 #!/bin/bash

DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xxl
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu
SUBJECT_FILE_PATH=data/mmlu/test

# MC_DROP_NUM=15
# EPOCH=2
OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/baseline/model/${MODEL}/mc/${MC_DROP_NUM}/epoch/${EPOCH}/${DATE}
mkdir -p ${OUTPUT_DIR}

for subcategory in physics chemistry biology computer_science math engineering history philosophy law politics culture economics geography psychology other business health
do 
    CUDA_VISIBLE_DEVICES=7 python run_mmlu_subcategory.py \
        --subcategory ${subcategory} \
        --model_name_or_path ${MODEL} \
        --dataset_name ${DATASET_NAME} \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_cache \
        --do_eval \
        --per_device_eval_batch_size 3
done

python print_output_subcategory.py --output_dir ${OUTPUT_DIR}
    