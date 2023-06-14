#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/mmlu/base/subcategory/run_mmlu_hard.sh
#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/mmlu/base/subcategory/run_mmlu_soft.sh


bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/train/trivia/xl/run_trivia_train.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/eval/nq/xl/run_nq_baseline.sh
#CUDA_VISIBLE_DEVICES=7 bash /data/syjeong/prompt_test/run/run.sh