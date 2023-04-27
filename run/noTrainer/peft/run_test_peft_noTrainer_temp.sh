DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
OUTPUT_DIR=./outputs/squad/noTrainer/temp/peft/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=1 python run_test_seq2seq_qa_noTrainer_peft.py \
    --model_name_or_path outputs/squad/noTrainer/temp/peft/2023_04_17/05_57_35 \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 12 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --do_eval \
    --do_peft \
    --max_eval_samples 10


    #--num_beams 1 \
    #--max_train_samples 10 \
    #--max_eval_samples 10