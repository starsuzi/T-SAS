DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-large
DATASET_NAME=trivia

for MC_DROP_NUM in 15
do
    for EPOCH in 5
    do
        for FILTER_THRES in 0.7
        do
            OUTPUT_DIR=./outputs/${DATASET_NAME}/test_time_tuning/model/${MODEL}/filter_thres/${FILTER_THRES}/no_lora/mc/${MC_DROP_NUM}/epoch/${EPOCH}/${DATE}
            mkdir -p ${OUTPUT_DIR}

            CUDA_VISIBLE_DEVICES=3 python run_tsas.py \
                --filter_thres ${FILTER_THRES} \
                --model_name_or_path ${MODEL} \
                --validation_file ./data/trivia/preprocessed/trivia_dev.json \
                --question_column question \
                --answer_column answers \
                --context_column context \
                --learning_rate 3e-5 \
                --max_seq_length 384 \
                --doc_stride 128 \
                --per_device_eval_batch_size 12 \
                --output_dir ${OUTPUT_DIR} \
                --overwrite_cache \
                --val_column 'validation' \
                --do_eval \
                --do_test_time_tuning \
                --mc_drop_num ${MC_DROP_NUM} \
                --test_time_tuning_epoch ${EPOCH}
        done
    done
done

            # --max_train_samples 5 \
            # --max_eval_samples 5 \
            # --max_test_time_tuning_samples 5