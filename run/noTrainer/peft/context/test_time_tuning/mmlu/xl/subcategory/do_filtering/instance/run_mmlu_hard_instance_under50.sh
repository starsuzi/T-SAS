DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu


for MC_DROP_NUM in 10 15 #10 15 # 5 15 25
do
    for EPOCH in 2 5 #3 5 #5 10 # 5 10 20
    do
        for FILTER_THRES in 0.7 0.9
        do
            OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/test_time_tuning/model/${MODEL}/subcategory/under_50/filter_thres/${FILTER_THRES}/lora/mc/${MC_DROP_NUM}/epoch/${EPOCH}/hard_label/${DATE}
            mkdir -p ${OUTPUT_DIR}

            #for subcategory in physics chemistry biology computer_science math engineering history philosophy law politics culture economics geography psychology other business health
            for subcategory in math health physics chemistry computer_science engineering philosophy law
            do 
                CUDA_VISIBLE_DEVICES=5 python run_mmlu_subcategory.py \
                    --subcategory ${subcategory} \
                    --filter_thres ${FILTER_THRES} \
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
done

#CUDA_VISIBLE_DEVICES=4 bash /data/syjeong/prompt_test/run/run.sh