#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME="natural_questions_val_fold_0" # "tfqa_val_fold_0", "triviaqa_val_fold_0"

python -m heads_selector.jsd_heads_selector.run \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME"