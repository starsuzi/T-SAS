#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/mmlu/base/subcategory/run_mmlu_hard.sh
#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/mmlu/base/subcategory/run_mmlu_soft.sh

#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/nq/xl/run_nq_baseline.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/train/nq/xl/run_nq_train.sh
CUDA_VISIBLE_DEVICES=3 bash /data/syjeong/prompt_test/run/run.sh