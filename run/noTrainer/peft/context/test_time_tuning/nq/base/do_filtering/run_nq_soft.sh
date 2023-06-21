DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-base
DATASET_NAME=nq

for MC_DROP_NUM in 15 #5 7 10
do
    for EPOCH in  5
    do
        OUTPUT_DIR=./outputs/${DATASET_NAME}/context/test_time_tuning/model/${MODEL}/orig_prompt/soft_label/no_lora/mc/${MC_DROP_NUM}/epoch/${EPOCH}/${DATE}
        mkdir -p ${OUTPUT_DIR}

        CUDA_VISIBLE_DEVICES=2 python run_squad.py \
            --model_name_or_path ${MODEL} \
            --validation_file /data/syjeong/prompt_test/data/nq/preprocessed/nq_dev.json \
            --do_soft_label \
            --question_column question \
            --answer_column answers \
            --context_column context \
            --learning_rate 3e-5 \
            --max_seq_length 384 \
            --doc_stride 128 \
            --per_device_eval_batch_size 24 \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_cache \
            --val_column 'validation' \
            --do_eval \
            --do_test_time_tuning \
            --mc_drop_num ${MC_DROP_NUM} \
            --test_time_tuning_epoch ${EPOCH}
    done
done

            #--num_beams 1 \
            # --max_train_samples 5 \
            # --max_eval_samples 50 \
            # --max_test_time_tuning_samples 50
