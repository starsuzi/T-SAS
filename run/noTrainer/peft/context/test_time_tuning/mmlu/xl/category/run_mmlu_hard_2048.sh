DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu


for MC_DROP_NUM in 2 # 5 10 15 # 25
do
    for EPOCH in 1 # 3 5 10 # 20
    do
        OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/context/test_time_tuning/model/${MODEL}/2048/category/lora/mc/${MC_DROP_NUM}/epoch/${EPOCH}/hard_label/${DATE}
        mkdir -p ${OUTPUT_DIR}

        for category in STEM humanities social_sciences other_business_health
        do 
            CUDA_VISIBLE_DEVICES=6 python run_mmlu_category.py \
                --category ${category} \
                --model_name_or_path ${MODEL} \
                --dataset_name ${DATASET_NAME} \
                --learning_rate 3e-5 \
                --max_seq_length 2048 \
                --doc_stride 128 \
                --output_dir ${OUTPUT_DIR} \
                --overwrite_cache \
                --train_peft_model \
                --val_column test \
                --do_eval \
                --do_test_time_tuning \
                --mc_drop_num ${MC_DROP_NUM} \
                --test_time_tuning_epoch ${EPOCH} \
                --per_device_eval_batch_size 2
        done

        python print_output_category.py --output_dir ${OUTPUT_DIR}

    done
done

CUDA_VISIBLE_DEVICES=3 bash /data/syjeong/prompt_test/run/run.sh