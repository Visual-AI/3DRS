import argparse
import torch
import torch.nn.functional as F

import os
import json
import ray
import time
import numpy as np
from tqdm import tqdm
import shortuuid
import fasteners

from transformers import AutoConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.video_utils import VideoProcessor, merge_video_dict

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

def average_coordinate_in_patch(world_coords, patch_size=27):
    V, H, W, D = world_coords.size() # D = 3

    world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
    world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]
    world_coords_avg = torch.nn.functional.avg_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
    patch_num = world_coords_avg.shape[-1]
    world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

    return world_coords_avg

def discrete_coords(world_coords, voxel_size=0.1):
    min_xyz_range = [-15, -15, -5]
    max_xyz_range = [15, 15, 5]

    min_xyz_range = torch.tensor(min_xyz_range).to(world_coords.device)
    max_xyz_range = torch.tensor(max_xyz_range).to(world_coords.device)

    world_coords = torch.maximum(world_coords, min_xyz_range)
    world_coords = torch.minimum(world_coords, max_xyz_range)
    world_coords_discrete = (world_coords - min_xyz_range) / voxel_size
    world_coords_discrete = world_coords_discrete.round()

    return world_coords_discrete.detach()

def ravel_hash_vec(arr):
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

def compute_voxel_similarity(features, voxel_ids):
    """
    Compute the average cosine similarity between features within the same voxel.

    Args:
        features (torch.Tensor): Input tensor of shape (1, L, C)
        voxel_ids (torch.Tensor): Voxel IDs tensor of shape (L,)

    Returns:
        torch.Tensor: Scalar tensor representing the final averaged similarity
    """
    assert features.shape[1] == voxel_ids.shape[0]
    # Remove batch dimension and normalize features
    features = features.squeeze(0)  # (L, C)
    features = F.normalize(features, p=2, dim=-1)

    # Compute cosine similarity matrix
    sim_matrix = torch.mm(features, features.T)  # (L, L)

    # Create masks for valid pairs
    same_voxel = voxel_ids.view(-1, 1) == voxel_ids.view(1, -1)
    upper_tri = torch.triu(torch.ones_like(same_voxel, dtype=torch.bool), diagonal=1)
    valid_mask = same_voxel & upper_tri

    # Extract valid similarities and their corresponding voxel groups
    rows, cols = torch.where(valid_mask)
    valid_sims = sim_matrix[rows, cols]
    group_ids = voxel_ids[rows]

    # Calculate average similarity per voxel
    if valid_sims.numel() == 0:
        return torch.tensor(0.0, device=features.device)

    unique_groups, inverse_indices = torch.unique(group_ids, return_inverse=True)
    sum_sims = torch.bincount(inverse_indices, weights=valid_sims, minlength=len(unique_groups))
    counts = torch.bincount(inverse_indices, minlength=len(unique_groups))
    voxel_avg = sum_sims / counts.clamp(min=1)  # Avoid division by zero

    # Return average over all voxels
    return voxel_avg.mean()

@ray.remote(num_gpus=1)
def eval_model(questions, args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    config = {}
    if args.lora_path is not None:
        config = AutoConfig.from_pretrained(args.lora_path)
        config = config.to_dict()
    elif args.overwrite_cfg:
        config.update({
            'tie_word_embeddings': False, 
            'use_cache': True, 
            "vocab_size": 151649
        })

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, overwrite_config=config)

    if args.lora_path is not None:
        from transformers import AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
        model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, args.lora_path, adapter_name="lora")
        model = model.merge_and_unload()
        state_dict = torch.load(os.path.join(args.lora_path, 'non_lora_trainables.bin'))
        msg = model.load_state_dict(state_dict, strict=False)
    
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "a")
    file_lock = fasteners.InterProcessLock(ans_file)

    video_processor = VideoProcessor(
        video_folder=args.video_folder,
        annotation_dir=args.embodiedscan_folder,
        # voxel_size=model.config.voxel_size,
        # min_xyz_range=model.config.min_xyz_range,
        # max_xyz_range=model.config.max_xyz_range,
        frame_sampling_strategy=args.frame_sampling_strategy,
    )
    
    ret = []
    inference_time = []
    for line in tqdm(questions):
        idx = line["id"]
        question_type = line["metadata"]["question_type"]
        dataset_name = line["metadata"]["dataset"]
        video_id = line["video"]


        if line["box_input"] is not None:
            # process box input
            box_input = line["box_input"][:3]
            # box_input = video_processor.discrete_point(
            #     box_input[:3],
            # )
            # center_point = [str(i) for i in center_point]
            # line['conversations'][0]["value"] = line['conversations'][0]["value"].replace('<coord>', ','.join(center_point))
            
            gt = line.get("annotations", [line["conversations"][1]["value"]])
            qs = line["conversations"][0]["value"]
            cur_prompt = args.extra_prompt + qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            args.conv_mode = "qwen_1_5"

            conv = conv_templates[args.conv_mode].copy()

            input_ids = preprocess_qwen([line["conversations"][0],{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
            img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

            video_dict = video_processor.process_3d_video(
                video_id,
                image_processor,
                force_sample=args.force_sample,
                frames_upbound=args.max_frame_num,
            )
            video_dict["box_input"] = box_input
            video_dict = merge_video_dict([video_dict])
            image_tensors = video_dict.pop('images').half().to(model.device)
            for k in video_dict:
                video_dict[k] = video_dict[k].half().to(model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                start_time = time.time()
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    modalities="video",
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=512,
                    use_cache=True,
                    video_dict=video_dict,
                )

                out = model(
                    input_ids=input_ids,
                    images=image_tensors,
                    modalities="video",
                    use_cache=True,
                    video_dict=video_dict,
                    return_dict=True,
                    output_hidden_states=True
                )
                world_coords = video_dict['world_coords'][0]
                world_coords = average_coordinate_in_patch(world_coords)
                world_coords_discrete = discrete_coords(world_coords)
                world_coords_discrete = world_coords_discrete.view(1, -1, 3)
                keys = ravel_hash_vec(world_coords_discrete) 
                keys_set = torch.unique(keys[0], return_inverse=True)
                #print(keys_set[1], keys_set[1].shape, keys_set[1].min(), keys_set[1].max())
                feature = out.hidden_states[-1][:,14:14+6720,:].view(32,14,15,-1)[:,:,:-1,:].contiguous().view(1,32*14*14,-1)
                avg_sim = compute_voxel_similarity(feature, keys_set[1])
                avg_sim = float(avg_sim.detach().cpu().item())
                # print('avg_sim', avg_sim)
                 
                inference_time.append(time.time() - start_time)
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        else:
            outputs = ""

        with file_lock:
            new_item = {
                    "dataset": dataset_name,
                    "sample_id": idx,
                    "prompt": cur_prompt,
                    "pred_response": outputs,
                    "gt_response": gt,
                    "model_id": model_name,
                    "question_type": question_type,
                    "scene": video_id,
                    "correspondence": avg_sim
            }
            ret.append(new_item)
            ans_file.write(json.dumps(new_item) + "\n")
            ans_file.flush()


    ans_file.close()
    return inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="data")
    parser.add_argument("--embodiedscan-folder", type=str, default="data/embodiedscan")
    parser.add_argument("--frame_sampling_strategy", type=str, default="uniform")
    parser.add_argument("--extra-prompt", type=str, default="The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--n_gpu", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    parser.add_argument("--max_frame_num", type=int, default=32)
    parser.add_argument("--force_sample", type=bool, default=True)
    parser.add_argument("--overwrite_cfg", type=bool, default=False)
    parser.add_argument("-n", type=int, default=-1)
    parser.add_argument("--lora-path", type=str, default=None)
    args = parser.parse_args()

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    if args.n != -1:
        questions = questions[:args.n]

    if os.path.exists(args.answer_file):
        print(f"The {args.answer_file} already exists!!!")
        exit()
    
    ray.init()
    features = []
    for i in range(args.n_gpu):
        features.append(eval_model.remote(questions[i::args.n_gpu], args))

    ret = ray.get(features)
    inference_time = []
    for item in ret:
        inference_time.extend(item)
    
    print(f"time: {np.mean(inference_time)}")


    
