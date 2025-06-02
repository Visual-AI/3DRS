#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="./ckpt/$1"
ANWSER_FILE="results/multi3drefer/$1.jsonl"


python3 llava/eval/model_multi3drefer.py \
    --model-path $CKPT \
    --video-folder ./data \
    --embodiedscan-folder data/embodiedscan \
    --n_gpu 8 \
    --question-file data/processed/multi3drefer_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --frame_sampling_strategy $2 \
    --max_frame_num $3 \
    --overwrite_cfg true

python llava/eval/eval_multi3drefer.py --input-file $ANWSER_FILE
