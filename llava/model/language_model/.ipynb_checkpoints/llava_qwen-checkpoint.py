#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config
from .qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM

import torch.distributed as dist
from deepspeed.comm import get_rank

def gather_loss(loss):
    # 将标量 loss 转为张量（确保设备一致）
    loss_tensor = torch.tensor([loss], device=torch.cuda.current_device())

    # 收集所有卡的 loss 张量
    world_size = dist.get_world_size()
    gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_losses, loss_tensor)

    # 转换为数值列表
    gathered_losses = [loss.item() for loss in gathered_losses]
    return gathered_losses


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if hasattr(config, "ground_head_type") and config.ground_head_type is not None:
            self.ground_head_type = config.ground_head_type
            if config.ground_head_type == "mlp":
                # self.ground_head = nn.Sequential(
                #     nn.Linear(config.hidden_size, config.ground_head_hidden_size),
                #     nn.ReLU(),
                #     nn.LayerNorm(config.ground_head_hidden_size),x
                #     nn.Linear(config.ground_head_hidden_size, 6)
                # )
                self.ground_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )
            elif config.ground_head_type == "score":
                self.ground_head_temperature = config.ground_head_temperature
                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_score = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1),
                )
            elif config.ground_head_type == "infonce":
                # self.ground_head_temperature = nn.Parameter(torch.tensor(config.ground_head_temperature))
                try:
                    self.ground_head_temperature = config.ground_head_temperature
                except:
                    self.ground_head_temperature = 0.07
                self.ground_head_zero_target = torch.nn.Parameter(torch.randn(config.hidden_size))

                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
            else:
                raise NotImplementedError
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        video_dict=None,
        use_object_proposals: bool = False,
        box_labels = None,
        img_pos_list = None,
        img_length_list = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, object_features, object_boxes, img_pos_list, img_length_list) = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids, 
                    position_ids, 
                    attention_mask, 
                    past_key_values, 
                    labels, 
                    images, 
                    modalities, 
                    image_sizes, 
                    video_dict,
                    use_object_proposals=use_object_proposals,
                )
            #print(img_pos_list, img_length_list)
        if use_object_proposals:
            return self.predict_box(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                object_features=object_features,
                object_boxes=object_boxes,
                box_labels=box_labels,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict
            )
            

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _, _, img_pos_list, img_length_list) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, video_dict=kwargs.get("video_dict", None))
            kwargs['img_pos_list'] = img_pos_list
            kwargs['img_length_list'] = img_length_list
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        img_pos_list = kwargs.pop("img_pos_list", None)
        img_length_list = kwargs.pop("img_length_list", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if img_pos_list is not None:
            inputs["img_pos_list"] = img_pos_list
        if img_length_list is not None:
            inputs["img_length_list"] = img_length_list

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    
    def average_coordinate_in_patch(self, world_coords, patch_size=27):
        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]
        world_coords_avg = torch.nn.functional.avg_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        patch_num = world_coords_avg.shape[-1]
        world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        return world_coords_avg

    def discrete_coords_new(self, world_coords, voxel_size=0.1):
        min_xyz_range = [-15, -15, -5]
        max_xyz_range = [15, 15, 5]

        min_xyz_range = torch.tensor(min_xyz_range).to(world_coords.device)
        max_xyz_range = torch.tensor(max_xyz_range).to(world_coords.device)

        world_coords = torch.maximum(world_coords, min_xyz_range)
        world_coords = torch.minimum(world_coords, max_xyz_range)
        world_coords_discrete = (world_coords - min_xyz_range) / voxel_size
        world_coords_discrete = world_coords_discrete.round()

        return world_coords_discrete.detach()

    def ravel_hash_vec(self, arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert len(arr.shape) == 3
        arr -= arr.min(1, keepdims=True)[0]
        arr_max = arr.max(1, keepdims=True)[0] + 1

        keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=arr.dtype).to(arr.device)

        # Fortran style indexing
        for j in range(arr.shape[2] - 1):
            keys += arr[..., j]
            keys *= arr_max[..., j + 1]
        keys += arr[..., -1]
        return keys

    def correspondence_loss(self, features, voxel_ids):
        """
        Compute correspondence and non-correspondence similarities across frames.

        Args:
            features (torch.Tensor): Input tensor of shape (1, L, C)
            voxel_ids (torch.Tensor): Voxel IDs tensor of shape (L,)

        Returns:
            tuple: (corr_avg, non_corr_avg) scalar tensors
        """
        # 输入验证和预处理
        assert features.shape[1] == voxel_ids.shape[0], "特征和voxel_id维度不匹配"
        assert features.shape[1] == 32 * 14 * 14, "特征维度不匹配"
        features = features.squeeze(0)  # (L, C)
        features = F.normalize(features, p=2, dim=-1)
        L = features.shape[0]
        
        # 计算帧索引 (假设特征按帧顺序排列)
        features_per_frame = 14 * 14  # 每帧196个voxel
        total_frames = 32
        frame_ids = torch.arange(L, device=features.device) // features_per_frame

        # 核心计算：余弦相似度矩阵 detach another one
        sim_matrix = torch.mm(features, features.detach().T)  # (L, L)

        # 生成基础掩码
        same_voxel = voxel_ids.view(-1, 1) == voxel_ids.view(1, -1)
        diff_frame = frame_ids.view(-1, 1) != frame_ids.view(1, -1)
        upper_tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=features.device), diagonal=1)

        # Correspondence掩码 (同一voxel不同帧)
        corr_mask = same_voxel & diff_frame & upper_tri
        
        # Non-correspondence掩码 (不同voxel不同帧)
        non_corr_mask = (~same_voxel) & diff_frame & upper_tri
        
        if corr_mask.sum() > 0:
            corr_sims = 1 - sim_matrix[corr_mask]
        else:
            corr_sims = 0.0
        if non_corr_mask.sum() > 0:
            non_corr_sims = sim_matrix[non_corr_mask]
        else:
            non_corr_sims = 0.0

        # 计算correspondence相似度（按voxel分组平均）
        '''
        corr_rows, corr_cols = torch.where(corr_mask)
        if corr_rows.numel() > 0:
            corr_sims = sim_matrix[corr_rows, corr_cols]
            correspondence_loss = 1 - corr_sims
        else:
            correspondence_loss = torch.tensor(0.0, device=features.device)
        '''
        return corr_sims.mean(), non_corr_sims.mean()
    
    def compute_correspondence(self, video_dict, hidden_states, voxel_size=0.1, img_pos_list=None, img_length_list=None):
        C = hidden_states.shape[-1]
        world_coords = video_dict['world_coords'][0]
        world_coords = self.average_coordinate_in_patch(world_coords)
        world_coords_discrete = self.discrete_coords_new(world_coords, voxel_size=voxel_size)
        world_coords_discrete = world_coords_discrete.view(1, -1, 3)
        keys = self.ravel_hash_vec(world_coords_discrete)
        keys_set = torch.unique(keys[0].long(), return_inverse=True)
        feature = hidden_states[:,img_pos_list[0]:img_pos_list[0]+img_length_list[0],:].view(32,14,15,C)[:,:,:-1,:].contiguous().view(1,32*14*14,C)
        corr_loss, non_corr_loss = self.correspondence_loss(feature, keys_set[1])
        return corr_loss, non_corr_loss
 
    def predict_box(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
        video_dict=None,
        object_features=None,
        object_boxes=None,
        box_labels=None,
        img_pos_list=None,
        img_length_list=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            img_pos_list=img_pos_list,
            img_length_list=img_length_list
        )

        hidden_states = outputs[0]
        ground_locations = (labels >= self.config.ground_token_ids[0]) & (labels <= self.config.ground_token_ids[-1])
        ground_hidden = hidden_states[ground_locations].squeeze(1)
        
        if self.ground_head_type == 'mlp':
            ground_hidden = self.ground_head(ground_hidden).squeeze(0) 
            scores = (ground_hidden * object_features).sum(dim=-1)
        elif self.ground_head_type == 'score':
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype)) # B, C
            query_feat = self.ground_head_query(ground_hidden) # 1, C
            # sim = (F.normalize(obj_feat) * F.normalize(query_feat)).sum(dim=-1)
            mul_feat = obj_feat * query_feat
            scores = self.ground_head_score(mul_feat) # B, 1
            scores = scores.squeeze(1)

        elif self.ground_head_type == "infonce":
            object_features = torch.cat([object_features, self.ground_head_zero_target.unsqueeze(0)], dim=0)
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype))
            query_feat = self.ground_head_query(ground_hidden)
            obj_feat = F.normalize(obj_feat)
            query_feat = F.normalize(query_feat)
            scores = (obj_feat * query_feat).sum(dim=-1)

        loss = None
        if box_labels is not None:
            if self.ground_head_type == "infonce":
                if len(box_labels[0]) == 0: # zero-target
                    box_labels[0].append(-1)
                logits = torch.exp(scores / self.ground_head_temperature)
                loss = - torch.log( logits[box_labels[0]].sum() / logits.sum())
                # negative_logits_sum = logits.sum() - logits[box_labels[0]].sum()
                # for idx in box_labels[0]:
                #     loss += - torch.log(logits[idx] / (negative_logits_sum + logits[idx]))
                # loss /= len(box_labels[0])
            else:
                bce_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                target = torch.zeros_like(scores)
                target[box_labels[0]] = 1
                weight = torch.ones_like(scores)
                if len(box_labels[0]) != 0:
                    weight[box_labels[0]] *= (scores.shape[0] - len(box_labels[0])) / len(box_labels[0])
                
                bce_loss = (bce_loss_fct(scores, target.detach()) * weight).mean()
                loss = bce_loss  
                # nce_loss = 0
                # logits = torch.exp(sim / self.ground_head_temperature)
                # negative_logits_sum = logits.sum() - logits[box_labels[0]].sum()
                # if len(box_labels[0]) != 0:
                #     for idx in box_labels[0]:
                #         nce_loss += - torch.log(logits[idx] / (negative_logits_sum + logits[idx]))
                #     nce_loss /= len(box_labels[0])
                # loss = bce_loss + nce_loss
        
        correspondence_loss, non_correspondence_loss = self.compute_correspondence(video_dict, hidden_states, voxel_size=0.1, img_pos_list=img_pos_list, img_length_list=img_length_list)
        loss += correspondence_loss + non_correspondence_loss
        gathered_correspondence_loss = gather_loss(correspondence_loss)
        gathered_non_correspondence_loss = gather_loss(non_correspondence_loss)
        if get_rank() == 0:
            avg_corr_loss = sum(gathered_correspondence_loss) / len(gathered_correspondence_loss)
            avg_non_corr_loss = sum(gathered_non_correspondence_loss) / len(gathered_non_correspondence_loss)
            print(f"Grounding Corresponding Loss: {avg_corr_loss:.4f} Non Corresponding Loss: {avg_non_corr_loss:.4f}")

        return loss, scores

        # loss = None
        # if box_labels is not None:
        #     ## BCE
        #     loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        #     target = torch.zeros_like(scores)
        #     target[box_labels[0]] = 1
        #     weight = torch.ones_like(scores)
        #     weight[box_labels[0]] *= scores.shape[0] - 1
        #     loss = (loss_fct(scores, target.detach()) * weight).mean()
        #     ## CE 
        #     # loss_fct = nn.CrossEntropyLoss()
        #     # loss = loss_fct(scores, box_labels[0]) / self.config.ground_loss_scale


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
