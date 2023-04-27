DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MC_DROP_NUM=10
OUTPUT_DIR=./outputs/covid/noTrainer/peft/mc/${MC_DROP_NUM}/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=2 python run_seq2seq_qa_noTrainer_peft_mc.py \
    --model_name_or_path outputs/covid/noTrainer/peft/mc/10/2023_04_20/08_53_14 \
    --dataset_name covid_qa_deepset \
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