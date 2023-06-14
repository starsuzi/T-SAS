DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=./outputs/nq/context/test_time_tuning/model/google/flan-t5-xxl/filter_thres/-1/orig_prompt/lora/mc/15/epoch/3/2023_06_12/16_56_07
DATASET_NAME=nq

OUTPUT_DIR=${MODEL}/eval/${DATE}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=7 python run_squad.py \
    --eval_peft_model \
    --model_name_or_path ${MODEL} \
    --validation_file /data/syjeong/prompt_test/data/nq/preprocessed/nq_dev.json \
    --question_column question \
    --answer_column answers \
    --context_column context \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 4 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'validation' \
    --do_eval


    #--num_beams 1 \
    # --max_train_samples 5 \
    # --max_eval_samples 5 \
    # --max_test_time_tuning_samples 5