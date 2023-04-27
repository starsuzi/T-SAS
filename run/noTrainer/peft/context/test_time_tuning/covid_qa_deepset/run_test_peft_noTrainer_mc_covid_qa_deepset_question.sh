DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=covid_qa_deepset

MC_DROP_NUM=10
EPOCH=2
OUTPUT_DIR=./outputs/${DATASET_NAME}/context/test_time_tuning/model/${MODEL}/mc/${MC_DROP_NUM}/epoch/${EPOCH}/with_multi_features_question/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=2 python run_seq2seq_qa_noTrainer_peft_mc.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --question_column question \
    --answer_column answers \
    --context_column context \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 4 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --train_peft_model \
    --val_column 'train' \
    --do_eval \
    --do_test_time_tuning \
    --mc_drop_num ${MC_DROP_NUM} \
    --test_time_tuning_epoch ${EPOCH} \
    --max_train_samples 5 \

    
    #--num_beams 1 \
    # --max_train_samples 5 \
    # --max_eval_samples 5 \
    # --max_test_time_tuning_samples 5