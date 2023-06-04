#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/mmlu/base/subcategory/run_mmlu_hard.sh
#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/mmlu/base/subcategory/run_mmlu_soft.sh

bash /data/syjeong/prompt_test/run/noTrainer/peft/context/train/squad/xl/run_squad.sh
CUDA_VISIBLE_DEVICES=5 bash /data/syjeong/prompt_test/run/run.sh