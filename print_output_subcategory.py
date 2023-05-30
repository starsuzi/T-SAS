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


subcategories_names = sorted(
    [
        f.split(".pkl")[0]
        for f in os.listdir(os.path.join(args.output_dir, 'subcategory_results'))
        if ".pkl" in f
    ]
)


all_cors = []
subcat_cors = {
    subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
}
cat_cors = {cat: [] for cat in categories}
logger.info('===============')
#import pdb; pdb.set_trace()
for subcategory in subcategories_names:
    with open(os.path.join(args.output_dir, 'subcategory_results', "{}.pkl".format(subcategory)), 'rb') as f:
        dict_subcategory_output = pickle.load(f)

        # assert
        assert subcategory == [i for i in dict_subcategory_output.keys()][0]
        cors = dict_subcategory_output[subcategory]['cors']
        #import pdb; pdb.set_trace()

        subcat_cors[subcategory].append(cors)
        for key in categories.keys():
            if subcategory in categories[key]:
                cat_cors[key].append(cors)
        all_cors.append(cors)

    #import pdb; pdb.set_trace()

for subcat in subcat_cors:
    #import pdb; pdb.set_trace()
    if subcat_cors[subcat] != []:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.4f} - {} - len: {}".format(subcat_acc, subcat, len(np.concatenate(subcat_cors[subcat]))))
        logger.info("Average accuracy {:.4f} - {} - len: {}".format(subcat_acc, subcat, len(np.concatenate(subcat_cors[subcat]))))

logger.info('===============')


for cat in cat_cors:
    if cat_cors[cat] != []:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.4f} - {} - len: {}".format(cat_acc, cat, len(np.concatenate(cat_cors[cat]))))
        logger.info("Average accuracy {:.4f} - {} - len: {}".format(cat_acc, cat, len(np.concatenate(cat_cors[cat]))))

logger.info('===============')


weighted_acc = np.mean(np.concatenate(all_cors))
print("Average accuracy: {:.4f} - len: {}".format(weighted_acc, len(np.concatenate(all_cors))))
logger.info("Average accuracy: {:.4f} - len: {}".format(weighted_acc, len(np.concatenate(all_cors))))