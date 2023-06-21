DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=bigscience/T0_3B
DATASET_NAME=sciq

OUTPUT_DIR=./outputs/${DATASET_NAME}/context/baseline/model/${MODEL}/orig_prompt/${DATE}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=5 python run_squad.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --question_column question \
    --answer_column correct_answer \
    --context_column support \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 8 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'test' \
    --do_eval \

    #--num_beams 1 \
    # --max_train_samples 5 \
    # --max_eval_samples 5 \
    # --max_test_time_tuning_samples 5