DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-small
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu
SUBJECT_FILE_PATH=data/mmlu/test


for MC_DROP_NUM in 3
do
    for EPOCH in 1
    do
        OUTPUT_DIR=./outputs/temp/${DATASET_NAME_FILENAME}/context/test_time_tuning/model/${MODEL}/lora/mc/${MC_DROP_NUM}/epoch/${EPOCH}/hard_label/${DATE}
        mkdir -p ${OUTPUT_DIR}

        for subject in $SUBJECT_FILE_PATH/*
        do 
            CUDA_VISIBLE_DEVICES=2 python run_mmlu_subject.py \
                --subject_file_path ${SUBJECT_FILE_PATH} \
                --subject ${subject} \
                --do_soft_label \
                --do_filtering \
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
                --per_device_eval_batch_size 4
        done

        for subject_result in $OUTPUT_DIR/subject_result/*
        do
            python print_output.py \
                --output_dir ${OUTPUT_DIR}
        done

    done
done