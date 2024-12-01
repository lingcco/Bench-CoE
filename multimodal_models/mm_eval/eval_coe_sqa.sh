#!/bin/bash

MODEL_PATH1="/mnt/data/zxj/checkpoints/TinyLLaVA-Phi-2-SigLIP-3.1B"
MODEL_PATH2="/mnt/data/zxj/checkpoints/Bunny-v1_0-3B"
MODEL_NAME="coe"
EVAL_DIR="/mnt/data/zxj/data"

python -m eval_coe_sqa \
    --model-path1 $MODEL_PATH1 \
    --model-path2 $MODEL_PATH2 \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode bunny

python TinyLLaVA_Factory/tinyllava/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_result.json