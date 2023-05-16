 #!/bin/bash

DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-base
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu
SUBJECT_FILE_PATH=data/mmlu/test

# MC_DROP_NUM=15
# EPOCH=2
OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/context/baseline/model/${MODEL}/mc/${MC_DROP_NUM}/epoch/${EPOCH}/${DATE}
mkdir -p ${OUTPUT_DIR}

for subject in $SUBJECT_FILE_PATH/*
do 
    CUDA_VISIBLE_DEVICES=4 python run_mmlu_subject.py \
        --subject_file_path ${SUBJECT_FILE_PATH} \
        --subject ${subject} \
        --model_name_or_path ${MODEL} \
        --dataset_name ${DATASET_NAME} \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_cache \
        --do_eval \
        --per_device_eval_batch_size 4
done

for subject_result in $OUTPUT_DIR/subject_result/*
do
    python print_output.py \
        --output_dir ${OUTPUT_DIR}
done
    