DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-large
DATASET_NAME=squad

OUTPUT_DIR=./outputs/${DATASET_NAME}/context/baseline/model/${MODEL}/${DATE}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=3 python run_tsas.py \
    --model_name_or_path ${MODEL} \
    --validation_file ./data/squad/preprocessed/squad_dev.json \
    --question_column question \
    --answer_column answers \
    --context_column context \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 24 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'validation' \
    --do_eval