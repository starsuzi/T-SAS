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


categories_names = sorted(
    [
        f.split(".pkl")[0]
        for f in os.listdir(os.path.join(args.output_dir, 'category_results'))
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
for category in categories_names:
    with open(os.path.join(args.output_dir, 'category_results', "{}.pkl".format(category)), 'rb') as f:
        dict_category_output = pickle.load(f)

        # assert
        assert category == [i for i in dict_category_output.keys()][0]
        cors = dict_category_output[category]['cors']
        #import pdb; pdb.set_trace()

        cat_cors[category].append(cors)

        all_cors.append(cors)

    #import pdb; pdb.set_trace()

logger.info('===============')

for cat in cat_cors:
    #import pdb; pdb.set_trace()
    if cat_cors[cat] != []:
        #import pdb; pdb.set_trace()
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.4f} - {} - len: {}".format(cat_acc, cat, len(np.concatenate(cat_cors[cat]))))
        logger.info("Average accuracy {:.4f} - {} - len: {}".format(cat_acc, cat, len(np.concatenate(cat_cors[cat]))))

logger.info('===============')


weighted_acc = np.mean(np.concatenate(all_cors))
print("Average accuracy: {:.4f} - len: {}".format(weighted_acc, len(np.concatenate(all_cors))))
logger.info("Average accuracy: {:.4f} - len: {}".format(weighted_acc, len(np.concatenate(all_cors))))