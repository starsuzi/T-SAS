DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=covid_qa_deepset

OUTPUT_DIR=./outputs/${DATASET_NAME}/context/baseline/model/orig_prompt/${MODEL}/${DATE}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=6 python run_squad.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --question_column question \
    --answer_column answers \
    --context_column context \
    --max_seq_length 2048 \
    --doc_stride 128 \
    --per_device_eval_batch_size 2 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'train' \
    --do_eval

    #--num_beams 1 \
    # --max_train_samples 5 \
    # --max_eval_samples 5 \
    # --max_test_time_tuning_samples 5