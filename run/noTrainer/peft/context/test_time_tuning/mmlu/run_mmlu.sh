DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu

MC_DROP_NUM=20
EPOCH=1
OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/context/test_time_tuning/model/${MODEL}/mc/${MC_DROP_NUM}/epoch/${EPOCH}/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=2 python run_mmlu.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --train_peft_model \
    --val_column test \
    --do_eval \
    --do_test_time_tuning \
    --mc_drop_num ${MC_DROP_NUM} \
    --test_time_tuning_epoch ${EPOCH} \
    --per_device_eval_batch_size 4 \
    