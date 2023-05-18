import json
import argparse
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

mapper = {
    'A' : 0,
    'B' : 1,
    'C' : 2,
    'D' : 3
}

parser = argparse.ArgumentParser(description="Finetune a transformers model on a QA task")
parser.add_argument("--json_dir", type=str, default='/data/syjeong/prompt_test/outputs/mmlu/context/confidence/model/google/flan-t5-xl/lora/mc/5/2023_05_18/19_21_47/json', help="Where to store the final model.")
args = parser.parse_args()

# Setup logging
logging.basicConfig(        
    filename=args.json_dir+'/logs.log', # 
    filemode='w',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    force=True
)

total_all_cors = []
total_filtered_cors = []

for file_name in sorted(os.listdir(args.json_dir)):
    subject = file_name.split(".json")[0]
    file_path = args.json_dir + '/' + file_name
    
    if '.json' in file_name:
        with open(file_path, 'r') as file:
            json_subject_data = json.load(file)
            subject_all_cors = []
            subject_filtered_cors = []

            lst_max_conf_per_mc = []
            for data in json_subject_data:
                for conf_per_mc in data['confidence']:
                    lst_max_conf_per_mc.append(max(conf_per_mc))
                    #import pdb; pdb.set_trace()
            
            median_conf = np.median(lst_max_conf_per_mc)
            #import pdb; pdb.set_trace()
            
            for subject_data in json_subject_data:
                pred = subject_data['max_vote']
                label = subject_data['gold_answer']
                #import pdb; pdb.set_trace()

                all_cor = pred == label
                subject_all_cors.append(all_cor)

                for tuple_conf_vote in zip(subject_data['confidence'], subject_data['max_vote']):
                    #import pdb; pdb.set_trace()
                    conf_for_winner = tuple_conf_vote[0][mapper[tuple_conf_vote[1]]]
                    #import pdb; pdb.set_trace()

                    if conf_for_winner >= median_conf: #0.5:
                        filtered_cor = pred == label
                        subject_filtered_cors.append(filtered_cor)


                    #import pdb; pdb.set_trace()

            total_all_cors = total_all_cors + subject_all_cors
            total_filtered_cors = total_filtered_cors + subject_filtered_cors
            #import pdb; pdb.set_trace()

            subject_all_acc = np.mean(subject_all_cors)
            subject_filtered_acc = np.mean(subject_filtered_cors)
            #import pdb; pdb.set_trace()

            logger.info("All Accuracy {:.4f} - {} - len: {}".format(subject_all_acc, subject, len(subject_all_cors)))
            logger.info("Filtered Accuracy {:.4f} - {} - len: {} - median: {}".format(subject_filtered_acc, subject, len(subject_filtered_cors), median_conf))
            logger.info("===========")


logger.info("******************")
logger.info("All Accuracy {:.4f} - Total".format(np.mean(total_all_cors)))
logger.info("Filtered Accuracy {:.4f} - Total".format(np.mean(total_filtered_cors)))
logger.info("******************")




        #import pdb; pdb.set_trace()


    


# subjects = sorted(
#     [
#         f.split("_test.csv")[0]
        
#         if ".json" in f
#     ]
# )
