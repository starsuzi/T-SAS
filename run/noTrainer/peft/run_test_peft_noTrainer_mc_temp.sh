DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MC_DROP_NUM=10
OUTPUT_DIR=./outputs/squad/noTrainer/temp/peft/mc/${MC_DROP_NUM}/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=1 python run_seq2seq_qa_noTrainer_peft_mc.py \
    --model_name_or_path 'google/flan-t5-xl' \
    --dataset_name squad \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 4 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --train_peft_model \
    --do_eval \
    --do_test_time_tuning \
    --mc_drop_num 10 \
    --test_time_tuning_epoch 2 \
    --max_train_samples 5 \
    --max_eval_samples 5 \
    --max_test_time_tuning_samples 5
    
    #--num_beams 1 \
    #--max_train_samples 10 \
    #--max_eval_samples 10