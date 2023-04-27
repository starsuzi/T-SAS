DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MC_DROP_NUM=10
OUTPUT_DIR=./outputs/squad/noTrainer/peft/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=1 python run_seq2seq_qa_noTrainer_peft_mc.py \
    --model_name_or_path /data/soyeong/prompt_test/outputs/squad/noTrainer/peft/mc/10/2023_04_20/04_55_42 \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 8 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --do_eval \
    --eval_peft_model \
    --max_train_samples 10 \
    
    #--num_beams 1 \
    #--max_train_samples 10 \
    #--max_eval_samples 10