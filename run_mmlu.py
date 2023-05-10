#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
import copy
#from utils_qa import *

import datasets
import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
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

# TODO: peft
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import PeftModel, PeftConfig

import pandas as pd
from mmlu_categories import subcategories, categories
from collections import Counter

import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

##
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


choices = ["A", "B", "C", "D"]


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a QA task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after "
            "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--do_eval", action="store_true", help="To do eval on the question answering model")
    parser.add_argument("--do_train", action="store_true", help="To do train on the question answering model")
    # data col
    parser.add_argument(
        "--train_column",
        type=str,
        default='train',
        help="The name of the train column in the datasets.",
    )
    parser.add_argument(
        "--val_column",
        type=str,
        default='validation',
        help="The name of the validation column in the datasets.",
    )
    parser.add_argument(
        "--test_column",
        type=str,
        default='test',
        help="The name of the test column in the datasets.",
    )
    # peft
    parser.add_argument("--eval_peft_model", action="store_true")
    parser.add_argument("--train_peft_model", action="store_true")
    # MC drop
    parser.add_argument("--do_test_time_tuning", action="store_true")
    parser.add_argument("--mc_drop_num", type=int)
    parser.add_argument('--test_time_tuning_epoch', type=int, default=2)
    parser.add_argument('--max_test_time_tuning_samples', type=int, default=None)
    #
    parser.add_argument('--without_multi_features', action="store_true")
    parser.add_argument("--ntrain", "-k", type=int, default=5)

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--val_max_answer_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--context_column",
        type=str,
        default='context',
        help="The name of the column in the datasets containing the contexts (for question answering).",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default='question',
        help="The name of the column in the datasets containing the questions (for question answering).",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default='answers',
        help="The name of the column in the datasets containing the answers (for question answering).",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    args = parser.parse_args()

    return args


# MC-drop
def run_mc_drop(model, tokenizer, input_ids, decoder_input_ids, args):
    logger.info("***** Running MC Drop *****")

    #model.eval()
    lst_mc_preds = []
    with torch.no_grad():
        for i in range(0, args.mc_drop_num):
            model.train()

            logits = model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids
            ).logits.flatten()

            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[0]],
                            logits[tokenizer("B").input_ids[0]],
                            logits[tokenizer("C").input_ids[0]],
                            logits[tokenizer("D").input_ids[0]],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

            logger.info('==========================================')
            logger.info(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
            logger.info('Prediction : ')
            logger.info(pred)
            logger.info(probs)

            lst_mc_preds.append(pred)

    return lst_mc_preds


# Majority Voting
def run_majority_vote(tokenizer, lst_mc_preds, args):
    logger.info("***** Running Majority Vote *****")

    mc_drop_num = args.mc_drop_num
    batch_size = len(lst_mc_preds[0])

    # convert list into dictionary
    dict_mc_freq_preds = Counter(lst_mc_preds)
    freq_pred = max(dict_mc_freq_preds, key=dict_mc_freq_preds.get)

    logger.info('===================================')
    logger.info('Max votes preds : ')
    logger.info(freq_pred)

    return freq_pred


def test_time_tuning(model, optimizer,lr_scheduler, tokenizer, test_time_tuning_dataloader, accelerator, args):

    logger.info("***** Running Test-time tuning *****")

    for epoch in range(0, args.test_time_tuning_epoch):
        model.train()
        for step, batch in enumerate(test_time_tuning_dataloader):
            batch['decoder_input_ids'] = None
            #import pdb; pdb.set_trace()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad() 

                logger.info("Test-time Loss:{} ".format(loss))   

        # save chenkpoint
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
  


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    #     (Pdb) df.iloc[idx]
    # 0    Find the degree for the given field extension ...
    # 1                                                    0
    # 2                                                    4
    # 3                                                    2
    # 4                                                    6
    # 5                                                    B
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    # import pdb; pdb.set_trace()
    # 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:'
    return prompt


def gen_prompt(train_df, subject, k=-1):
    # (Pdb) train_df
    #                                                 0           1             2            3            4  5
    # 0  Find all c in Z_3 such that Z_3[x]/(x^2 + c) i...           0             1            2            3  B
    # 1  Statement 1 | If aH is an element of a factor ...  True, True  False, False  True, False  False, True  B
    # 2  Statement 1 | Every element of a group generat...  True, True  False, False  True, False  False, True  C
    # 3  Statement 1| Every function from a finite set ...  True, True  False, False  True, False  False, True  A
    # 4            Find the characteristic of the ring 2Z.           0             3           12           30  A

    # (Pdb) prompt
    # 'The following are multiple choice questions (with answers) about  abstract algebra.\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nStatement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: B\n\nStatement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: C\n\nStatement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: A\n\nFind the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: A\n\n'
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    #import pdb; pdb.set_trace()
    return prompt


def eval(args, subject, model, optimizer, lr_scheduler, tokenizer, dev_df, test_df, accelerator):

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    def preprocess_function(examples):
        inputs = examples['input_ids']
        pred_labels = examples['labels']

        padding = "max_length" if args.pad_to_max_length else False
        model_inputs = tokenizer(inputs, max_length=args.max_seq_length, padding=padding, truncation=True)

        # Tokenize targets with text_target=...
        labels = tokenizer(text_target=pred_labels, max_length=args.max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        #import pdb; pdb.set_trace()

        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs

    
    model.eval()
    # test time tuning
    if args.do_test_time_tuning:
        test_time_tuning_examples = {}
        lst_input = []
        lst_pred_label = []

        # rows with long prompt
        lst_long_row = []

        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

            while input_ids.shape[-1] > args.max_seq_length and k >=0 : #2048:
                k -= 1
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            
            if k < 0 :
                # lst_long_row.append(i)
                continue            
            
            lst_input.append(prompt)
            #import pdb; pdb.set_trace()        
            decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
            decoder_input_ids = model._shift_right(decoder_input_ids)

            # gold label
            label = test_df.iloc[i, test_df.shape[1] - 1]

            # mc drop
            lst_mc_preds = run_mc_drop(model, tokenizer, input_ids, decoder_input_ids, args)
            #import pdb; pdb.set_trace()

            # majority vote
            pred_label = run_majority_vote(tokenizer, lst_mc_preds, args)
            lst_pred_label.append(pred_label)

            logger.info('Gold Answer')
            logger.info(label)


        # test_time_tuning
        test_time_tuning_examples['input_ids'] = lst_input
        test_time_tuning_examples['labels'] = lst_pred_label

        test_time_tuning_examples = datasets.Dataset.from_dict(test_time_tuning_examples)

        test_time_tuning_dataset = test_time_tuning_examples.map(
                        preprocess_function,
                        batched=True,
                        num_proc=args.preprocessing_num_workers,
                        # remove_columns=test_time_tuning_examples.column_names,
                        load_from_cache_file=not args.overwrite_cache,
                    )


        test_time_tuning_dataloader = DataLoader(
            test_time_tuning_dataset, collate_fn=data_collator, shuffle=False, batch_size=args.per_device_eval_batch_size
        )
    
        del test_time_tuning_dataset
        del test_time_tuning_examples

        #import pdb; pdb.set_trace()
        test_time_tuning_dataloader = accelerator.prepare(test_time_tuning_dataloader)
        test_time_tuning(model, optimizer, lr_scheduler, tokenizer, test_time_tuning_dataloader, accelerator, args)
        #import pdb; pdb.set_trace()

        del test_time_tuning_dataloader
        
    # actual inference
    cors = []
    all_probs = []

    answers = choices[: test_df.shape[1] - 2]

    # rows with long prompt
    lst_long_row = []

    model.eval()
    with torch.no_grad():
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
 

            while input_ids.shape[-1] > args.max_seq_length and k >=0 : #2048:
                k -= 1
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
       
            if k < 0 :
                lst_long_row.append(i)
                continue
                
            label = test_df.iloc[i, test_df.shape[1] - 1]

            decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
            decoder_input_ids = model._shift_right(decoder_input_ids)

            logits = model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids
            ).logits.flatten()

            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer("A").input_ids[0]],
                            logits[tokenizer("B").input_ids[0]],
                            logits[tokenizer("C").input_ids[0]],
                            logits[tokenizer("D").input_ids[0]],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
            #import pdb; pdb.set_trace()
            cor = pred == label
            cors.append(cor)
            all_probs.append(probs)

        acc = np.mean(cors)
        cors = np.array(cors)

        all_probs = np.array(all_probs)
        print("Average accuracy {:.3f} - {}".format(acc, subject))
        logger.info("Average accuracy {:.3f} - {}".format(acc, subject))

    # Drop rows with long prompts
    test_df = test_df.drop(labels=lst_long_row, axis=0)

    return cors, acc, all_probs, test_df
    
def load_model(args, test_df):
    # Load pretrained model and tokenizer
    # eval peft model
    if args.eval_peft_model:
        peft_model_id = args.model_name_or_path
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    else:
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        if args.model_name_or_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForSeq2SeqLM.from_config(config)

        # TODO: peft
        if args.train_peft_model:
            peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))
    # if model.config.decoder_start_token_id is None:
    #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(test_df) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.test_time_tuning_epoch * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    return model, tokenizer, optimizer, lr_scheduler

def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )


    # Make one log on every process with the configuration for debugging.
    # TODO
    # Setup logging
    logging.basicConfig(        
        filename=args.output_dir+'/logs.log', # 
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True
    )

    #logger.info(accelerator.state, main_process_only=False)
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # data
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.dataset_name, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, "results_{}".format(args.model_name_or_path))):
        os.makedirs(os.path.join(args.output_dir, "results_{}".format(args.model_name_or_path)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.dataset_name, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.dataset_name, "test", subject + "_test.csv"), header=None
        )

        # model for each_subject
        model, tokenizer, optimizer, lr_scheduler = load_model(args, test_df)
        #import pdb; pdb.set_trace()

        # Prepare everything with our `accelerator`.
        model, optimizer, dev_df, test_df, lr_scheduler = accelerator.prepare(
                    model, optimizer, dev_df, test_df, lr_scheduler
                )
        # import pdb; pdb.set_trace()
        cors, acc, probs, test_df = eval(args, subject, model, optimizer, lr_scheduler, tokenizer, dev_df, test_df, accelerator)
        
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model_name_or_path)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model_name_or_path, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.output_dir, "results_{}".format(args.model_name_or_path), "{}.csv".format(subject)
            ),
            index=None,
        )

        model.cpu()
        
        del model
        del optimizer
        del lr_scheduler
        
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()



    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
        logger.info("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        logger.info("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    logger.info("Average accuracy: {:.3f}".format(weighted_acc))







if __name__ == "__main__":
    main()





