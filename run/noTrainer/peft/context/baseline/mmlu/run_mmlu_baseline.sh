DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu

# MC_DROP_NUM=15
# EPOCH=2
OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/context/baseline/model/${MODEL}/mc/${MC_DROP_NUM}/epoch/${EPOCH}/${DATE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python run_mmlu.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --do_eval \
    --per_device_eval_batch_size 4 \
    