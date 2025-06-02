#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="./ckpt/$1"
ANWSER_FILE="results/scanqa/$1.jsonl"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 llava/eval/model_scanqa.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 8 \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --question-file data/processed/scanqa_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --overwrite_cfg true

python llava/eval/eval_scanqa.py --input-file $ANWSER_FILE
