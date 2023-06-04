DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=squad

EPOCH=2

OUTPUT_DIR=./outputs/${DATASET_NAME}/context/train/reproduce/model/${MODEL}/lora/epoch/${EPOCH}/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=4 python run_squad.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --question_column question \
    --answer_column answers \
    --context_column context \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --train_peft_model \
    --val_column 'validation' \
    --do_eval \
    --do_train \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size 6 \



    #--num_beams 1 \
    # --max_train_samples 5 \
    # --max_eval_samples 5 \
    # --max_test_time_tuning_samples 5

CUDA_VISIBLE_DEVICES=4 bash /data/syjeong/prompt_test/run/run.sh