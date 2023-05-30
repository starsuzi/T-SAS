DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=google/flan-t5-xl
DATASET_NAME=data/mmlu
DATASET_NAME_FILENAME=mmlu

EPOCH=3

OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/context/test_time_tuning/model/${MODEL}/label/category/lora/epoch/${EPOCH}/hard_label/${DATE}
mkdir -p ${OUTPUT_DIR}

for category in STEM humanities social_sciences other_business_health
do 
    CUDA_VISIBLE_DEVICES=6 python run_mmlu_category_label.py \
        --category ${category} \
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
        --test_time_tuning_epoch ${EPOCH} \
        --per_device_eval_batch_size 12
done

python print_output_category.py --output_dir ${OUTPUT_DIR}


####################################################################

EPOCH=5

OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/context/test_time_tuning/model/${MODEL}/label/category/lora/epoch/${EPOCH}/hard_label/${DATE}
mkdir -p ${OUTPUT_DIR}

for category in STEM humanities social_sciences other_business_health
do 
    CUDA_VISIBLE_DEVICES=6 python run_mmlu_category_label.py \
        --category ${category} \
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
        --test_time_tuning_epoch ${EPOCH} \
        --per_device_eval_batch_size 12
done

python print_output_category.py --output_dir ${OUTPUT_DIR}



# CUDA_VISIBLE_DEVICES=3 bash /data/syjeong/prompt_test/run/run.sh


####################################################################

EPOCH=10

OUTPUT_DIR=./outputs/${DATASET_NAME_FILENAME}/context/test_time_tuning/model/${MODEL}/label/category/lora/epoch/${EPOCH}/hard_label/${DATE}
mkdir -p ${OUTPUT_DIR}

for category in STEM humanities social_sciences other_business_health
do 
    CUDA_VISIBLE_DEVICES=6 python run_mmlu_category_label.py \
        --category ${category} \
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
        --test_time_tuning_epoch ${EPOCH} \
        --per_device_eval_batch_size 12
done

python print_output_category.py --output_dir ${OUTPUT_DIR}
