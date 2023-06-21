DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=trivia
PROMPT_STYPE=cq_prompt

OUTPUT_DIR=./outputs/${DATASET_NAME}/context/baseline/model/${MODEL}/${PROMPT_STYPE}/${DATE}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=4 python run_squad_prompt.py \
    --prompt_style ${PROMPT_STYPE} \
    --model_name_or_path ${MODEL} \
    --validation_file /data/syjeong/prompt_test/data/trivia/preprocessed/trivia_dev.json \
    --question_column question \
    --answer_column answers \
    --context_column context \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 12 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'validation' \
    --do_eval

    #--num_beams 1 \
    # --max_train_samples 5 \
    # --max_eval_samples 5 \
    # --max_test_time_tuning_samples 5