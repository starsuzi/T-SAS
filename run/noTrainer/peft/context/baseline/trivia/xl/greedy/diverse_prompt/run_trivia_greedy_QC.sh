DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=trivia
PROMPT_STYPE=QC_prompt

for MC_DROP_NUM in 1
do
    for EPOCH in 5
    do
        for FILTER_THRES in -1
        do
            OUTPUT_DIR=./outputs/${DATASET_NAME}/context/baseline/greedy/model/${MODEL}/${PROMPT_STYPE}/lora/epoch/${EPOCH}/${DATE}
            mkdir -p ${OUTPUT_DIR}

            CUDA_VISIBLE_DEVICES=6 python run_squad_prompt_greedy.py \
                --prompt_style ${PROMPT_STYPE} \
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
                --train_peft_model \
                --val_column 'validation' \
                --do_eval \
                --do_test_time_tuning \
                --mc_drop_num ${MC_DROP_NUM} \
                --test_time_tuning_epoch ${EPOCH}
        done
    done
done

            #--num_beams 1 \
            # --max_train_samples 5 \
            # --max_eval_samples 5 \
            # --max_test_time_tuning_samples 5