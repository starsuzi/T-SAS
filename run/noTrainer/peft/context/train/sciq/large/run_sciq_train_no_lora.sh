DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-large
DATASET_NAME=sciq

for EPOCH in 1
do

    OUTPUT_DIR=./outputs/${DATASET_NAME}/context/train/model/${MODEL}/no_lora/orig_prompt/lora/epoch/${EPOCH}/${DATE}
    mkdir -p ${OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=3 python run_squad.py \
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
        --train_column 'train' \
        --val_column 'test' \
        --do_eval \
        --do_train \
        --num_train_epochs ${EPOCH} \
        --per_device_train_batch_size 24
done


        #--num_beams 1 \
        # --max_train_samples 5 \
        # --max_eval_samples 5 \
        # --max_test_time_tuning_samples 5