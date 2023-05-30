DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu


for MC_DROP_NUM in 5 10 15 # 25
do
    for EPOCH in 3 5 # 10 # 20
    do
        OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/context/test_time_tuning/model/${MODEL}/category/do_filtering/lora/mc/${MC_DROP_NUM}/epoch/${EPOCH}/soft_label/softLabelDivider/${DATE}
        mkdir -p ${OUTPUT_DIR}

        for category in STEM humanities social_sciences other_business_health
        do 
            CUDA_VISIBLE_DEVICES=7 python run_mmlu_category_softLabelDivider.py \
                --category ${category} \
                --do_filtering \
                --do_soft_label \
                --model_name_or_path ${MODEL} \
                --dataset_name ${DATASET_NAME} \
                --learning_rate 3e-5 \
                --max_seq_length 384 \
                --doc_stride 128 \
                --output_dir ${OUTPUT_DIR} \
                --overwrite_cache \
                --train_peft_model \
                --val_column test \
                --do_eval \
                --do_test_time_tuning \
                --mc_drop_num ${MC_DROP_NUM} \
                --test_time_tuning_epoch ${EPOCH} \
                --per_device_eval_batch_size 12
        done

        python print_output_category.py --output_dir ${OUTPUT_DIR}
            
    done
done

# CUDA_VISIBLE_DEVICES=7 bash /data/syjeong/prompt_test/run/run.sh