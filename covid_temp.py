import os
os.environ['TRANSFORMERS_CACHE'] = '/data/syjeong/cache'
import numpy as np
import datasets
import transformers
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint

raw_datasets = load_dataset('covid_qa_deepset')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')

max_len = 0
lst_current_len = []
for c in tqdm(raw_datasets['train']['context']):
    current_len = len(tokenizer(c).input_ids)
    lst_current_len.append(current_len)
    #import pdb; pdb.set_trace()
    if max_len < current_len:
        max_len = current_len
        #print(max_len)

import pdb; pdb.set_trace()