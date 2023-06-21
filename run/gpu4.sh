# bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/no_context/run_trivia_baseline.sh
# bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/squad_dpr/xl/no_context/run_squad_dpr_baseline.sh
# bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_smcho_prompt.sh
# bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_CQ_prompt.sh
# bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_cq_prompt.sh
# bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_qc_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_QC_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_smcho_reverse_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_smcho_article_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_article_answer_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_article_answer_reverse_prompt.sh

bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_smcho_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_smcho_reverse_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_smcho_article_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_article_answer_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_article_answer_reverse_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_CQ_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_cq_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_qc_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_QC_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/baseline/trivia/xl/run_trivia_baseline_smcho_article_reverse_prompt.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/trivia/xl/do_filtering/instance/run_trivia_instance_filter_smcho_article_reverse_prompt.sh

CUDA_VISIBLE_DEVICES=4 bash /data/syjeong/prompt_test/run/run.sh