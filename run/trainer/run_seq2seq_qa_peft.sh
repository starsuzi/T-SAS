DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
OUTPUT_DIR=./outputs/debug_seq2seq_squad/peft/flan-t5-xl/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=1 python run_seq2seq_qa_peft.py \
  --model_name_or_path 'google/flan-t5-xl' \
  --dataset_name squad \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ${OUTPUT_DIR} \
  --overwrite_output_dir \
  --overwrite_cache \
  --predict_with_generate \