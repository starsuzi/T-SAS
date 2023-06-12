DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=trivia

for EPOCH in 1 2
do

    OUTPUT_DIR=./outputs/${DATASET_NAME}/context/train/model/${MODEL}/orig_prompt/lora/epoch/${EPOCH}/${DATE}
    mkdir -p ${OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=1 python run_squad.py \
        --model_name_or_path ${MODEL} \
        --validation_file ./data/trivia/preprocessed/trivia_dev_all.json \
        --train_file ./data/trivia/preprocessed/trivia_train_all.json \
        --answer_column answers \
        --context_column context \
        --learning_rate 3e-5 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_cache \
        --train_peft_model \
        --train_column 'train' \
        --val_column 'validation' \
        --do_eval \
        --do_train \
        --num_train_epochs ${EPOCH} \
        --per_device_train_batch_size 8
done


        #--num_beams 1 \
        # --max_train_samples 5 \
        # --max_eval_samples 5 \
        # --max_test_time_tuning_samples 5