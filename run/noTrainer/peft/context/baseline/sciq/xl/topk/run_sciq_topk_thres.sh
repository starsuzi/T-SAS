DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=sciq

for MC_DROP_NUM in 10 #7 10 15
do
    for EPOCH in 5
    do
        for FILTER_THRES in -1
        do
            OUTPUT_DIR=./outputs/${DATASET_NAME}/context/baseline/topk/model/${MODEL}/filter_thres/${FILTER_THRES}/orig_prompt/lora/topk_num/${MC_DROP_NUM}/epoch/${EPOCH}/${DATE}
            mkdir -p ${OUTPUT_DIR}

            CUDA_VISIBLE_DEVICES=6 python run_squad_topk.py \
                --filter_thres ${FILTER_THRES} \
                --model_name_or_path ${MODEL} \
                --dataset_name ${DATASET_NAME} \
                --question_column question \
                --answer_column correct_answer \
                --context_column support \
                --learning_rate 3e-5 \
                --max_seq_length 384 \
                --doc_stride 128 \
                --per_device_topk_batch_size 1 \
                --per_device_eval_batch_size 12 \
                --output_dir ${OUTPUT_DIR} \
                --overwrite_cache \
                --train_peft_model \
                --val_column 'test' \
                --do_eval \
                --do_test_time_tuning \
                --test_time_tuning_epoch ${EPOCH} \
                --mc_drop_num ${MC_DROP_NUM}
        done
    done
done

            #--num_beams 1 \
            # --max_train_samples 5 \
            # --max_eval_samples 5 \
            # --max_test_time_tuning_samples 5