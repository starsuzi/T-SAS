DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu


for MC_DROP_NUM in 5 7 10 #10 15 # 5 15 25
do
    for EPOCH in 2 3 4 5 #3 5 #5 10 # 5 10 20
    do
        OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/test_time_tuning/model/${MODEL}/subcategory/do_filtering/lora/mc/${MC_DROP_NUM}/epoch/${EPOCH}/soft_label/${DATE}
        mkdir -p ${OUTPUT_DIR}

        for subcategory in physics chemistry biology computer_science math engineering history philosophy law politics culture economics geography psychology other business health
        do 
            CUDA_VISIBLE_DEVICES=6 python run_mmlu_subcategory.py \
                --subcategory ${subcategory} \
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

        python print_output_subcategory.py --output_dir ${OUTPUT_DIR}

    done
done

#CUDA_VISIBLE_DEVICES=5 bash /data/syjeong/prompt_test/run/run.sh