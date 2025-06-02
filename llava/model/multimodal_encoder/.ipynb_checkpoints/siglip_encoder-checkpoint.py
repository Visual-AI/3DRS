"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from llava.utils import rank0_print


class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        # self.msg_tokens = nn.Parameter(torch.zeros(1, 16, self.embed_dim))

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        # msg_tokens = self.msg_tokens.repeat(embeddings.shape[0],1,1)
        # embeddings = torch.cat([msg_tokens, embeddings],dim=1) # 16+L
        return embeddings

class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

def voxelize_and_average(feat_xyz, features):
    batch_size, N, _ = feat_xyz.size()
    C = features.size(2)
    voxel_indices = feat_xyz.long() # (B, N, 3)
    # voxel_indices = (feat_xyz / voxel_size).floor().long()  # (B, N, 3)
    min_indices = voxel_indices.min(dim=1, keepdim=True)[0]  # (B, 1, 3)
    max_indices = voxel_indices.max(dim=1, keepdim=True)[0]  # (B, 1, 3)
    grid_shape = (max_indices - min_indices + 1).squeeze(1)  # (B, 3)

    voxel_features = []
    for b in range(batch_size):
        # 获取第b个batch的数据
        indices_b = voxel_indices[b] - min_indices[b, 0]  # 调整索引从零开始
        features_b = features[b]  # (N, C)
        grid_shape_b = grid_shape[b]  # (3,)

        # 计算平面索引
        strides_b = torch.tensor([grid_shape_b[1]*grid_shape_b[2],
                                  grid_shape_b[2],
                                  1], dtype=torch.long, device=feat_xyz.device)  # (3,)
        indices_flat_b = (indices_b * strides_b).sum(dim=1)  # (N,)

        # 初始化累加器
        num_voxels_b = grid_shape_b.prod().item()
        voxel_feature_sums = torch.zeros((num_voxels_b, C), device=features_b.device, dtype=features_b.dtype)
        voxel_counts = torch.zeros((num_voxels_b,), device=features_b.device, dtype=torch.long)

        # 累加特征和计数
        voxel_feature_sums.index_add_(0, indices_flat_b, features_b)
        voxel_counts.index_add_(0, indices_flat_b, torch.ones_like(indices_flat_b))

        # 计算均值
        non_zero_mask = voxel_counts > 0
        # print('non_zero_mask:', non_zero_mask.sum()/(grid_shape.prod().item()))
        voxel_features_mean = torch.zeros_like(voxel_feature_sums)
        voxel_features_mean[non_zero_mask] = voxel_feature_sums[non_zero_mask] / voxel_counts[non_zero_mask].unsqueeze(-1)

        # 重塑为栅格形状
        voxel_features_mean = voxel_features_mean.view(*grid_shape_b.tolist(), C)

        voxel_features.append(voxel_features_mean)
    voxel_features = torch.stack(voxel_features, dim=0)

    return voxel_features

def map_voxel_features_to_points(feat_xyz, voxel_features):
    batch_size, N, _ = feat_xyz.size()
    per_point_features = []

    voxel_indices = feat_xyz.long()  # (B, N, 3)

    # voxel_indices = (feat_xyz / voxel_size).floor().long()  # (B, N, 3)
    min_indices = voxel_indices.min(dim=1, keepdim=True)[0]  # (B, 1, 3)

    for b in range(batch_size):
        # 调整索引从零开始
        indices_b = voxel_indices[b] - min_indices[b, 0]  # (N, 3)
        voxel_features_b = voxel_features[b]  # (D, H, W, C)

        # 获取每个点对应的栅格特征
        point_features_b = voxel_features_b[indices_b[:, 0], indices_b[:, 1], indices_b[:, 2]]  # (N, C)

        per_point_features.append(point_features_b)

    per_point_features = torch.stack(per_point_features, dim=0)  # (B, N, C)

    return per_point_features


class CLIP_3DAdapter(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.activation_fn = ACT2FN[config.hidden_act]
        self.adapter_linear_down = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
        self.adapter_conv = nn.Conv3d(self.embed_dim // 4, self.embed_dim // 4, kernel_size=(3, 3, 3), padding=1, groups=self.embed_dim // 4, bias=False)
        self.adapter_smooth_conv = nn.Conv2d(self.embed_dim // 4, self.embed_dim // 4, kernel_size=(3, 3), padding=1, groups=self.embed_dim // 4, bias=False)
        self.adapter_linear_up = nn.Linear(self.embed_dim // 4, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        coords=None
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
        """
        
        residual = hidden_states
        coords = coords[0]
        V, L, C = hidden_states.shape
        H, W = coords.shape[1], coords.shape[2]
        assert coords.shape[0] == V and H*W == L and coords.shape[3] == 3
        hidden_states_down = self.adapter_linear_down(hidden_states)
        C_down = hidden_states_down.shape[-1]
        hidden_states_down = hidden_states_down.view(1, V*L, C_down)
        coords = coords.view(1, V*L, 3)
        hidden_states_voxelised = voxelize_and_average(coords, hidden_states_down) # B, D, H, W, C_down
        
        hidden_states_voxelised = hidden_states_voxelised.permute(0, 4, 1, 2, 3).contiguous() # B, C_down, D, H, W
        hidden_states_adapted = self.adapter_conv(hidden_states_voxelised) # B, C_down, D, H, W
        hidden_states_adapted = self.activation_fn(hidden_states_adapted)
        hidden_states_adapted = hidden_states_adapted.permute(0, 2, 3, 4, 1).contiguous() # B, D, H, W, C_down
        
        hidden_states_adapted = map_voxel_features_to_points(coords, hidden_states_adapted) # B, N, C_down
        hidden_states_adapted = hidden_states_adapted.view(V, H, W, C_down) # V, H, W, C_down
        hidden_states_adapted = hidden_states_adapted.permute(0, 3, 1, 2).contiguous() # V, C_down, H, W
        hidden_states_adapted_smooth = self.adapter_smooth_conv(hidden_states_adapted).permute(0, 2, 3, 1).contiguous() # V, H, W, C_down

        hidden_states_adapted_smooth = hidden_states_adapted_smooth.view(1, V*H*W, C_down)
        hidden_states_adapted_up = self.adapter_linear_up(hidden_states_adapted_smooth).view(V, H*W, C)

        outputs = residual + hidden_states_adapted_up

        return outputs

class CLIP_3D_Reuse(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.activation_fn = ACT2FN[config.hidden_act]
        self.adapter_linear_down = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
        self.adapter_conv = nn.Conv3d(self.embed_dim // 4, self.embed_dim // 4, kernel_size=(3, 3, 3), padding=1, groups=self.embed_dim // 4, bias=False)
        self.adapter_smooth_conv = nn.Conv2d(self.embed_dim // 4, self.embed_dim // 4, kernel_size=(3, 3), padding=1, groups=self.embed_dim // 4, bias=False)
        self.adapter_linear_up = nn.Linear(self.embed_dim // 4, self.embed_dim, bias=False)

    def forward(
        self,
        video_tensor: torch.Tensor,
        coord_info=None
    ) -> Tuple[torch.FloatTensor]:
        """
        将多视角视频张量根据坐标信息转换为体素表示，整体处理后再逆映射回视频序列

        参数：
        - video_tensor: (V, L, C)，多视角视频序列的特征张量
        - coord_info: (V, H, W, 3)，视频序列的坐标信息，其中L=H*W
        - xxx_function: 对体素特征进行处理的函数

        返回：
        - processed_video_tensor: (V, L, C)，处理后的视频特征张量
        """
        V, L, C = video_tensor.size()
        H, W = coord_info.size(1), coord_info.size(2)
        assert L == H * W, "L should be equal to H * W"

        # 将坐标信息从 (V, H, W, 3) 重塑为 (V, L, 3)
        coord_info = coord_info.view(V, L, 3)

        # 将所有视角的数据合并
        features_all = video_tensor.view(V * L, C)  # (V*L, C)
        coord_all = coord_info.view(V * L, 3)       # (V*L, 3)

        # 将坐标离散化为体素索引
        voxel_indices = torch.floor(coord_all).long()  # (V*L, 3)

        # 调整体素索引从零开始
        min_indices = voxel_indices.min(dim=0, keepdim=True)[0]  # (1, 3)
        voxel_indices -= min_indices  # 体素索引现在从零开始

        # 计算全局体素网格形状
        max_indices = voxel_indices.max(dim=0, keepdim=True)[0]  # (1, 3)
        grid_shape = (max_indices - min_indices + 1).squeeze(0)  # (3,)

        # 计算用于展平体素索引的步长
        strides = torch.tensor([grid_shape[1]*grid_shape[2],
                                grid_shape[2],
                                1], dtype=torch.long, device=voxel_indices.device)  # (3,)

        # 展平体素索引
        indices_flat = (voxel_indices * strides).sum(dim=1)  # (V*L,)

        # 初始化累加器
        num_voxels = grid_shape.prod().item()
        voxel_feature_sums = torch.zeros((num_voxels, C), device=features_all.device, dtype=features_all.dtype)
        voxel_counts = torch.zeros((num_voxels,), device=features_all.device, dtype=torch.long)

        # 累加特征和计数
        voxel_feature_sums.index_add_(0, indices_flat, features_all)
        voxel_counts.index_add_(0, indices_flat, torch.ones_like(indices_flat))

        # 计算每个体素的平均特征
        non_zero_mask = voxel_counts > 0
        voxel_features = torch.zeros_like(voxel_feature_sums)
        voxel_features[non_zero_mask] = voxel_feature_sums[non_zero_mask] / voxel_counts[non_zero_mask].unsqueeze(-1)

        # 对非空体素特征进行整体处理
        voxel_features_processed = voxel_features.clone()
        voxel_features_processed[non_zero_mask] = xxx_function(voxel_features[non_zero_mask])

        # 将处理后的体素特征映射回原始位置
        processed_features_all = voxel_features_processed[indices_flat]  # (V*L, C)

        # 重塑为 (V, L, C)
        processed_video_tensor = processed_features_all.view(V, L, C)

        return processed_video_tensor

# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig, layer_idx):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # self.activation_fn = ACT2FN[config.hidden_act]
        # self.pooling_3d = nn.AvgPool3d(kernel_size=(3,3,3), stride=(3,3,3), padding=0)
        # self.adapter = CLIP_3DAdapter(config)
        # self.vision_adapter_linear_down = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
        # self.vision_adapter_linear_up = nn.Linear(self.embed_dim // 4, self.embed_dim, bias=False)
        # self.vision_adapter_conv = nn.Conv3d(self.embed_dim // 4, self.embed_dim // 4, kernel_size=(3, 3, 3), padding=1, groups=self.embed_dim // 4, bias=False)
        # self.vision_adapter_linear_down_2 = nn.Linear(self.embed_dim, self.embed_dim // 4, bias=False)
        # self.vision_adapter_linear_up_2 = nn.Linear(self.embed_dim // 4, self.embed_dim, bias=False)

    def obtain_voxelizaton_feat(self, video_tensor, coord_info):
        V, L, C = video_tensor.size()
        coord_info = coord_info[0]
        H, W = coord_info.size(1), coord_info.size(2)
        assert L == H * W, "L should be equal to H * W"

        # 将坐标信息从 (V, H, W, 3) 重塑为 (V, L, 3)
        coord_info = coord_info.view(V, L, 3)

        # 将所有视角的数据合并
        features_all = video_tensor.contiguous().view(V * L, C)  # (V*L, C)
        coord_all = coord_info.view(V * L, 3)       # (V*L, 3)

        # 将坐标离散化为体素索引
        voxel_indices = torch.floor(coord_all).long()  # (V*L, 3)

        # 调整体素索引从零开始
        min_indices = voxel_indices.min(dim=0, keepdim=True)[0]  # (1, 3)
        max_indices = voxel_indices.max(dim=0, keepdim=True)[0]  # (1, 3)

        voxel_indices -= min_indices  # 体素索引现在从零开始

        # 计算全局体素网格形状
        
        grid_shape = (max_indices - min_indices + 1).squeeze(0)  # (3,)

        # 计算用于展平体素索引的步长
        strides = torch.tensor([grid_shape[1]*grid_shape[2],
                                grid_shape[2],
                                1], dtype=torch.long, device=voxel_indices.device)  # (3,)

        # 展平体素索引
        indices_flat = (voxel_indices * strides).sum(dim=1)  # (V*L,)

        # 初始化累加器
        num_voxels = grid_shape.prod().item()
        voxel_feature_sums = torch.zeros((num_voxels, C), device=features_all.device, dtype=features_all.dtype)
        voxel_counts = torch.zeros((num_voxels,), device=features_all.device, dtype=torch.long)

        # 累加特征和计数
        voxel_feature_sums.index_add_(0, indices_flat, features_all)
        voxel_counts.index_add_(0, indices_flat, torch.ones_like(indices_flat))

        # 计算每个体素的平均特征
        non_zero_mask = voxel_counts > 0
        voxel_features = torch.zeros_like(voxel_feature_sums)
        voxel_features[non_zero_mask] = voxel_feature_sums[non_zero_mask] / voxel_counts[non_zero_mask].unsqueeze(-1)
        
        voxel_features = voxel_features.view(*grid_shape.tolist(), C).permute(3, 0, 1, 2).contiguous().unsqueeze(0) # B C H W Z
        voxel_features = self.pooling_3d(voxel_features).view(C, -1)
        
        non_zero_mask = torch.any(voxel_features, dim=0)
        voxel_features = voxel_features[:,non_zero_mask].permute(1,0).contiguous()
       
        return voxel_features

    def make_voxel_aggreagation_mask(self, feat, img_len, voxel_len):
        mask = torch.zeros((feat.size(0), 1, img_len+voxel_len, img_len+voxel_len), device=feat.device)
        mask[0,0,img_len:, :] = -1e9
        return mask

    def spatial_temporal_adaper(self, video_tensor, coord_info):
        V, L, C = video_tensor.size()
        coord_info = coord_info[0]
        H, W = coord_info.size(1), coord_info.size(2)
        assert L == H * W, "L should be equal to H * W"
        video_tensor_dc = self.activation_fn(self.vision_adapter_linear_down(video_tensor))
        video_tensor_dc = video_tensor_dc.view(1, V, H, W, C//4).permute(0, 4, 1, 2, 3).contiguous()
        video_tensor_dc = self.vision_adapter_conv(video_tensor_dc)
        video_tensor_dc = video_tensor_dc.permute(0, 2, 3, 4, 1).contiguous()
        video_tensor_dc = video_tensor_dc.view(V, L, C//4)
        video_tensor_up = self.vision_adapter_linear_up(video_tensor_dc)
        return video_tensor_up
    

    def attn_reuse(self, video_tensor, coord_info):
        V, L, C = video_tensor.size()
        coord_info = coord_info[0]
        H, W = coord_info.size(1), coord_info.size(2)
        assert L == H * W, "L should be equal to H * W"

        # 将坐标信息从 (V, H, W, 3) 重塑为 (V, L, 3)
        coord_info = coord_info.view(V, L, 3)

        # 将所有视角的数据合并
        features_all = video_tensor.view(V * L, C)  # (V*L, C)
        coord_all = coord_info.view(V * L, 3)       # (V*L, 3)

        # 将坐标离散化为体素索引
        voxel_indices = torch.floor(coord_all).long()  # (V*L, 3)

        # 调整体素索引从零开始
        min_indices = voxel_indices.min(dim=0, keepdim=True)[0]  # (1, 3)
        max_indices = voxel_indices.max(dim=0, keepdim=True)[0]  # (1, 3)

        voxel_indices -= min_indices  # 体素索引现在从零开始

        # 计算全局体素网格形状
        
        grid_shape = (max_indices - min_indices + 1).squeeze(0)  # (3,)


        # 计算用于展平体素索引的步长
        strides = torch.tensor([grid_shape[1]*grid_shape[2],
                                grid_shape[2],
                                1], dtype=torch.long, device=voxel_indices.device)  # (3,)

        # 展平体素索引
        indices_flat = (voxel_indices * strides).sum(dim=1)  # (V*L,)

        # 初始化累加器
        num_voxels = grid_shape.prod().item()
        voxel_feature_sums = torch.zeros((num_voxels, C), device=features_all.device, dtype=features_all.dtype)
        voxel_counts = torch.zeros((num_voxels,), device=features_all.device, dtype=torch.long)

        # 累加特征和计数
        voxel_feature_sums.index_add_(0, indices_flat, features_all)
        voxel_counts.index_add_(0, indices_flat, torch.ones_like(indices_flat))

        # 计算每个体素的平均特征
        non_zero_mask = voxel_counts > 0
        voxel_features = torch.zeros_like(voxel_feature_sums)
        voxel_features[non_zero_mask] = voxel_feature_sums[non_zero_mask] / voxel_counts[non_zero_mask].unsqueeze(-1)

        # 对非空体素特征进行整体处理
        voxel_features_processed = voxel_features.clone()
        feat = voxel_features[non_zero_mask].unsqueeze(0)
        with torch.no_grad():
            attn_voxel_feat, _ = self.self_attn(
                feat,
                attention_mask=None,
                output_attentions=False
            )
        attn_voxel_feat = self.vision_adapter_linear_up(self.activation_fn(self.vision_adapter_linear_down(attn_voxel_feat)))
        # print('self.adapter_linear_up:', self.vision_adapter_linear_up.weight[0,0])
        # print('self.adapter_linear_down:', self.vision_adapter_linear_down.weight[0,0])
        voxel_features_processed[non_zero_mask] = attn_voxel_feat.squeeze(0)

        # 将处理后的体素特征映射回原始位置
        processed_features_all = voxel_features_processed[indices_flat]  # (V*L, C)

        # 重塑为 (V, L, C)
        processed_video_tensor = processed_features_all.view(V, L, C)

        return processed_video_tensor

    def MsgShift(self, feature, msg_token_length=16, shift_info=[0,0,-1,1,-2,2]):
        msg_tokens, patch_feat = feature[:,:msg_token_length], feature[:,msg_token_length:]
        # print('feature sizes:', msg_tokens.shape, patch_feat.shape)
        msg_tokens = msg_tokens.chunk(len(shift_info), dim=-1)
        msg_tokens = [torch.roll(msg, shifts=roll, dims=0) for msg, roll in zip(msg_tokens, shift_info)]
        msg_tokens = torch.cat(msg_tokens, dim=-1)
        feature = torch.cat([msg_tokens, patch_feat], dim=1)
        return feature
  
    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        coords=None
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states


        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = residual + hidden_states

        # hidden_states = self.MsgShift(hidden_states)        

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        coords = None
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    coords
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    coords=coords
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        coords = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            coords=coords
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = last_hidden_state[:,16:,:] if last_hidden_state.shape[1] > 729 else last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        coords= None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            coords=coords
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map, local_files_only=True)

        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images, coords=None):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, coords=coords)
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, coords=coords)
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            if image_features.shape[-2] == 729 + 16: 
                image_features = image_features[:,16:,:]
            assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size
