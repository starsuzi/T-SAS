DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
OUTPUT_DIR=./outputs/squad/noTrainer/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=2 python run_seq2seq_qa_noTrainer.py \
    --model_name_or_path google/flan-t5-base \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_train_batch_size 12 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --do_train \
    --do_eval \


    #--num_beams 1 \
    #--max_train_samples 10 \
    #--max_eval_samples 10