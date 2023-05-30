import pickle
import argparse
import logging
import os
import numpy as np
from mmlu_categories import subcategories, categories

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Finetune a transformers model on a QA task")
parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

args = parser.parse_args()

# TODO
# Setup logging
logging.basicConfig(        
    filename=args.output_dir+'/logs.log', # 
    filemode='a',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    force=True
)


    
subjects = sorted(
    [
        f.split(".pkl")[0]
        for f in os.listdir(os.path.join(args.output_dir, 'subject_results'))
        if ".pkl" in f
    ]
)



all_cors = []
subcat_cors = {
    subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
}
cat_cors = {cat: [] for cat in categories}

logger.info('===============')
for subject in subjects:
    with open(os.path.join(args.output_dir, 'subject_results', "{}.pkl".format(subject)), 'rb') as f:
        dict_subject_output = pickle.load(f)

        # assert
        assert subject == [i for i in dict_subject_output.keys()][0]
        cors = dict_subject_output[subject]['cors']

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    #import pdb; pdb.set_trace()

for subcat in subcat_cors:
    #import pdb; pdb.set_trace()
    if subcat_cors[subcat] != []:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.4f} - {}".format(subcat_acc, subcat))
        logger.info("Average accuracy {:.4f} - {}".format(subcat_acc, subcat))

for cat in cat_cors:
    if cat_cors[cat] != []:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.4f} - {}".format(cat_acc, cat))
        logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))

weighted_acc = np.mean(np.concatenate(all_cors))
print("Average accuracy: {:.4f}".format(weighted_acc))
logger.info("Average accuracy: {:.4f}".format(weighted_acc))