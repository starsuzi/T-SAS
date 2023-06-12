#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/mmlu/base/subcategory/run_mmlu_hard.sh
#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/mmlu/base/subcategory/run_mmlu_soft.sh

bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter.sh
CUDA_VISIBLE_DEVICES=7 bash /data/syjeong/prompt_test/run/run.sh