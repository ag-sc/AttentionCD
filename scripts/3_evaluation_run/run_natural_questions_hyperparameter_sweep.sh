#!/bin/bash
# Note: Choose the best hyper-parameter and comment out rest of the values in the list.

DATA_PATH="data/natural_questions"  

MODEL_PATHS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

ALPHA_LIST=(
    0.1
)

THRESHOLD_PERCENTILES=(
    1
    5
    10
    15
    20
    25
    50
    75
    100
)

MODES=(
    "jsd"
    "ds"
)


for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    # find the corresponding head map path and layers to ablate list based on the model name
    if [ "$MODEL_NAME" == "Llama-3.2-1B-Instruct" ]; then
        LAYERS_TO_ABLATE_LIST=(
            "range(0, 16, 1)"
            "range(0, 8, 1)"
            "range(7, 16, 1)"
        )
    elif [ "$MODEL_NAME" == "Llama-3.2-3B-Instruct" ]; then
        LAYERS_TO_ABLATE_LIST=(
            "range(0, 28, 1)"
            "range(0, 14, 1)"
            "range(13, 28, 1)"
        )
    elif [ "$MODEL_NAME" == "Llama-3.1-8B-Instruct" ]; then
        LAYERS_TO_ABLATE_LIST=(
            "range(0, 32, 1)"
            "range(0, 16, 1)"
            "range(15, 32, 1)"
        )
    else
        echo "Unknown MODEL_NAME: $MODEL_NAME"
        exit 1
    fi     

    for THRESHOLD_PERCENTILE in "${THRESHOLD_PERCENTILES[@]}"; do
        for MODE in "${MODES[@]}"; do
            if [ "$MODE" == "jsd" ]; then
                MAP_NAME="js_divs_per_layer/heads_dict_js_divs_top_${THRESHOLD_PERCENTILE}_${MODEL_NAME}"
                LAYER_TO_HEAD_MAP_PATH="head_divs_results_natural_questions/${MAP_NAME}.json"
            elif [ "$MODE" == "ds" ]; then
                MAP_NAME="ds_per_layer/heads_dict_ds_top_${THRESHOLD_PERCENTILE}_${MODEL_NAME}"
                LAYER_TO_HEAD_MAP_PATH="head_divs_results_natural_questions/${MAP_NAME}.json"
            else
                echo "Unknown MODE: $MODE"
                exit 1
            fi
      
            for LAYERS_TO_ABLATE in "${LAYERS_TO_ABLATE_LIST[@]}"; do
                for ALPHA in "${ALPHA_LIST[@]}"; do
                    echo "Running MODEL_NAME: $MODEL_NAME THRESHOLD_PERCENTILE: $THRESHOLD_PERCENTILE LAYERS_TO_ABLATE: $LAYERS_TO_ABLATE ALPHA: $ALPHA"
                    IFS=', ' read -r start end step <<< "$(echo "$LAYERS_TO_ABLATE" | sed 's/range(\(.*\))/\1/')"
                    out_path = "result_AttentionCD_cd_${ALPHA}_layers_${start}_${end}.json"

                    python -m benchmark.triviaqa_natural_questions \
                        --model-name "$MODEL_PATH" \
                        --data-path "$DATA_PATH" \
                        --mode att_cd \
                        --layer_indices "$LAYERS_TO_ABLATE" \
                        --head_indices "$LAYER_TO_HEAD_MAP_PATH" \
                        --cd_temperature "$ALPHA" \
                        --dataset_name "natural_questions" \
                        --output-path "results/${MODE}/results_natural_questions_val_post_all/$MODEL_NAME/threshold_$THRESHOLD_PERCENTILE/${out_path}"                    
            done
        done
    done
done
