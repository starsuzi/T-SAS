DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=covid_qa_deepset

MC_DROP_NUM=10
EPOCH=2
OUTPUT_DIR=./outputs/${DATASET_NAME}/context/test_time_tuning/model/${MODEL}/mc/${MC_DROP_NUM}/epoch/${EPOCH}/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python run_seq2seq_qa_noTrainer_peft_mc.py \
    --model_name_or_path outputs/covid_qa_deepset/context/test_time_tuning/model/google/flan-t5-xl/mc/10/epoch/2/2023_04_21/18_56_17 \
    --dataset_name ${DATASET_NAME} \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 8 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'train' \
    --do_eval \
    --eval_peft_model \
    --max_train_samples 1 \
    
    #--num_beams 1 \
    #--max_train_samples 10 \
    #--max_eval_samples 10