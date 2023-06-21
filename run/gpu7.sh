#bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/sciq/large/do_filtering/instance/run_sciq_instance_filter_no_lora.sh
# bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/nq/large/do_filtering/instance/run_nq_instance_filter_no_lora.sh
# bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/squad_dpr/large/do_filtering/instance/run_squad_dpr_instance_filter_no_lora.sh
# bash /data/syjeong/prompt_test/run/gpu7_base.sh
bash /data/syjeong/prompt_test/run/noTrainer/peft/context/test_time_tuning/squad_dpr/xl/do_filtering/instance/run_squad_dpr_instance_filter.sh
CUDA_VISIBLE_DEVICES=7 bash /data/syjeong/prompt_test/run/run.sh