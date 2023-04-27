DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
OUTPUT_DIR=./outputs/squad_v2/noTrainer/temp/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python run_seq2seq_qa_noTrainer.py \
    --model_name_or_path outputs/squad_v2/noTrainer/2023_04_12/18_15_08 \
    --dataset_name squad_v2 \
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
    --do_eval \
    --version_2_with_negative \
    --max_eval_samples 30 \
    

    #--num_beams 1 \