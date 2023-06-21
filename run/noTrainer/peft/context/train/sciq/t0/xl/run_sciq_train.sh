DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=bigscience/T0_3B
DATASET_NAME=sciq

for EPOCH in 1 2
do
    OUTPUT_DIR=./outputs/${DATASET_NAME}/context/train/model/${MODEL}/orig_prompt/lora/epoch/${EPOCH}/${DATE}
    mkdir -p ${OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=5 python run_squad.py \
        --model_name_or_path ${MODEL} \
        --dataset_name ${DATASET_NAME} \
        --question_column question \
        --answer_column correct_answer \
        --context_column support \
        --learning_rate 3e-5 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_cache \
        --train_peft_model \
        --val_column 'test' \
        --do_eval \
        --do_train \
        --num_train_epochs ${EPOCH} \
        --per_device_train_batch_size 6 \

        #--num_beams 1 \
        # --max_train_samples 5 \
        # --max_eval_samples 5 \
        # --max_test_time_tuning_samples 5
done
