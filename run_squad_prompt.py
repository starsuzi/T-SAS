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
os.environ['TRANSFORMERS_CACHE'] = '/data/syjeong/cache'
import random
from pathlib import Path
from typing import List, Optional, Tuple
import copy
#from utils_qa import *
import pickle

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

from torch.nn import CrossEntropyLoss

from utils_prompt import *



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
    # cot
    parser.add_argument("--do_cot", action="store_true")
    # peft
    parser.add_argument("--eval_peft_model", action="store_true")
    parser.add_argument("--train_peft_model", action="store_true")
    # MC drop
    parser.add_argument("--do_test_time_tuning", action="store_true")
    parser.add_argument("--mc_drop_num", type=int)
    parser.add_argument('--test_time_tuning_epoch', type=int, default=2)
    parser.add_argument('--max_test_time_tuning_samples', type=int, default=None)
    # filter_thres
    parser.add_argument("--filter_thres", type=float, default=-1)
    # soft label
    parser.add_argument("--do_soft_label", action="store_true")
    #
    parser.add_argument("--prompt_style", type=str)
    #
    parser.add_argument('--without_multi_features', action="store_true")
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
    # TODO 
    parser.add_argument(
            "--max_test_time_train_steps",
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

    # Sanity checks
    if args.dataset_name is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    device = accelerator.device

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


    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
        #import pdb; pdb.set_trace()
        raw_datasets = load_dataset(extension, data_files=data_files)
        
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # load model and tokenizer
    model, tokenizer = load_model(args)

        
    if args.do_train:
        if args.train_column not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets[args.train_column]

        # TODO if there's no 'id' column
        if args.dataset_name == 'sciq':
            # TODO make 'id'
            lst_id = [str(i) for i in range(len(train_dataset['question']))]
            max_id_length = len(lst_id[-1])
            lst_str_id = [('_')*(max_id_length-len(i))+i for i in lst_id]

            train_dataset = train_dataset.add_column('id', lst_str_id)
            #import pdb; pdb.set_trace()

        if args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(args.max_train_samples))


        # Create train feature from dataset
        with accelerator.main_process_first():
            # column_names = raw_datasets["train"].column_names
            train_dataset = train_dataset.map(
                preprocess_features_function, 
                fn_kwargs={'args':args, 'raw_datasets':raw_datasets, 'tokenizer': tokenizer},
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            if args.max_train_samples is not None:
                # Number of samples might increase during Feature Creation, We select only specified max samples
                train_dataset = train_dataset.select(range(args.max_train_samples))
    
    
    if args.val_column not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_examples = raw_datasets[args.val_column]

    # TODO if there's no 'id' column
    if args.dataset_name == 'sciq':
        # TODO make 'id'
        lst_id = [str(i) for i in range(len(eval_examples['question']))]
        max_id_length = len(lst_id[-1])
        lst_str_id = [('_')*(max_id_length-len(i))+i for i in lst_id]

        eval_examples = eval_examples.add_column('id', lst_str_id)
        

    if args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(args.max_eval_samples))
    # Validation Feature Creation
    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(
            preprocess_features_function, 
            fn_kwargs={'args':args, 'raw_datasets':raw_datasets, 'tokenizer': tokenizer},
            batched=True,
            num_proc=args.preprocessing_num_workers,
            #remove_columns=column_names,
            remove_columns=eval_examples.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if args.max_eval_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))


    if args.do_test_time_tuning:
        test_time_tuning_examples = raw_datasets[args.val_column]

        # TODO if there's no 'id' column
        if args.dataset_name == 'sciq':
            # TODO make 'id'
            lst_id = [str(i) for i in range(len(test_time_tuning_examples['question']))]
            max_id_length = len(lst_id[-1])
            lst_str_id = [('_')*(max_id_length-len(i))+i for i in lst_id]

            test_time_tuning_examples = test_time_tuning_examples.add_column('id', lst_str_id)

            
        if args.max_test_time_tuning_samples is not None:
            # We will select sample from whole data
            test_time_tuning_examples = test_time_tuning_examples.select(range(args.max_test_time_tuning_samples))

        # test_time_tuning Feature Creation
        with accelerator.main_process_first():
            test_time_tuning_dataset = test_time_tuning_examples.map(
                preprocess_features_function, 
                fn_kwargs={'args':args, 'raw_datasets':raw_datasets, 'tokenizer': tokenizer},
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=test_time_tuning_examples.column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on eval dataset",
            )
        if args.max_test_time_tuning_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            test_time_tuning_dataset = test_time_tuning_dataset.select(range(args.max_test_time_tuning_samples))


    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )


    if args.do_train:
        train_dataset_for_model = train_dataset.remove_columns(["example_id", "offset_mapping"])
        train_dataloader = DataLoader(
            train_dataset_for_model, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )


    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])        
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    if args.do_test_time_tuning:       
        test_time_tuning_dataset_for_model = test_time_tuning_dataset.remove_columns(["example_id", "offset_mapping"])        
        test_time_tuning_dataloader = DataLoader(
            test_time_tuning_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
       

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


    # if args.do_train:
    #     # Scheduler and math around the number of training steps.
    #     overrode_max_train_steps = False
    #     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    #     if args.max_train_steps is None:
    #         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #         overrode_max_train_steps = True

    #     lr_scheduler_train = get_scheduler(
    #         name=args.lr_scheduler_type,
    #         optimizer=optimizer,
    #         num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
    #         num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    #     )


    # Prepare everything with our `accelerator`.
    model, optimizer, eval_dataloader = accelerator.prepare(
        model, optimizer, eval_dataloader
    )

    if args.do_test_time_tuning:
        test_time_tuning_dataloader = accelerator.prepare(
            test_time_tuning_dataloader
        )
    if args.do_train:
        train_dataloader = accelerator.prepare(
            train_dataloader #, lr_scheduler_train
        )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # if args.do_train:
    #     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    #     if overrode_max_train_steps:
    #         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     # Afterwards we recalculate our number of training epochs
    #     args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # if args.do_test_time_tuning:
    #     num_update_steps_per_epoch = math.ceil(len(test_time_tuning_dataloader) / args.gradient_accumulation_steps)

    #     if overrode_max_train_steps:
    #         args.max_test_time_train_steps = args.test_time_tuning_epoch * num_update_steps_per_epoch
    #     # Afterwards we recalculate our number of training epochs
    #     args.test_time_tuning_epoch = math.ceil(args.max_test_time_train_steps / num_update_steps_per_epoch)


    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    # Metric
    metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    # Train!
    if args.do_train:
        
        args.max_train_steps, args.num_train_epochs, lr_scheduler_train = prepare_scheduler(args, accelerator, train_dataloader, optimizer, args.max_train_steps, args.num_train_epochs)

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler_train.step()
                    optimizer.zero_grad()

                    # logger.info("Loss:{} ".format(loss))

                    # We keep track of the loss at each epoch
                    total_loss = total_loss + loss.cpu().detach().float()
                #import pdb; pdb.set_trace()
                logger.info(tokenizer.batch_decode(batch["input_ids"][:1], skip_special_tokens=True))

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            logger.info("Epoch %d Loss:{} ".format(total_loss / len(train_dataloader)), epoch) 

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )


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


    # MC-drop
    def run_mc_drop(model, tokenizer, batch, gen_kwargs, args):
        logger.info("***** Running MC Drop *****")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        #model.eval()
        lst_generated_tokens = []
        with torch.no_grad():
            for i in range(0, args.mc_drop_num):
                model.train()

                outputs = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                    #return_dict_in_generate=True,
                    #output_scores=True
                )
                # import pdb; pdb.set_trace()
                # # calculate prob
                # transition_scores = model.compute_transition_scores(
                #     outputs.sequences, outputs.scores, normalize_logits=True
                # )

                # transition_scores = accelerator.gather_for_metrics(transition_scores)
                # transition_scores = transition_scores.cpu()

                # output_length = batch["input_ids"].shape[1] + np.sum(transition_scores.numpy() < 0, axis=1)
                # probabilities = torch.exp(transition_scores.sum(axis=1) / (output_length))

                # generated_tokens
                generated_tokens = outputs#.sequences
                generated_tokens = accelerator.gather_for_metrics(generated_tokens)
                generated_tokens = generated_tokens.cpu().numpy()

                # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
                # encoder-decoder models, like BART or T5.
                # delete bos token
                # generated_tokens = generated_tokens[:, 1:] 
                input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                generated_tokens = generated_tokens[:, input_length:]

                # gold labels
                gold_labels = batch['labels']
                gold_labels = gold_labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    gold_labels = np.where(gold_labels != -100, gold_labels, tokenizer.pad_token_id)
                
                logger.info('==========================================')
                logger.info(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False))
                logger.info('Prediction : ')
                logger.info(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
                #logger.info('Probabilities : ')
                #logger.info(probabilities)
                logger.info('Answer : ')
                logger.info(tokenizer.batch_decode(gold_labels, skip_special_tokens=True))

                # delete tokenizer.pad_token_id
                generated_tokens = np.array([[(t if t != tokenizer.pad_token_id else -100) for t in tok] for tok in generated_tokens])
                
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                lst_generated_tokens.append(generated_tokens)

        return lst_generated_tokens

    #
    def convert_to_arr(lst_mc_generated_tokens, args):
        mc_drop_num = args.mc_drop_num
        batch_size = len(lst_mc_generated_tokens[0])

        # find max seq length to make an array
        max_seq_len = max([x.shape[1] for x in lst_mc_generated_tokens])
        
        lst_pad_mc_generated_tokens = []
        for x in lst_mc_generated_tokens: 
            # append if not requiring padding
            if x.shape[1] == max_seq_len:
                lst_pad_mc_generated_tokens.append(x)
            # else pad the seq
            else:
                arr_pad_row_col = np.full((batch_size, max_seq_len), -100)
                
                if args.ignore_pad_token_for_loss:
                    arr_pad_row = np.full(max_seq_len - x.shape[1], -100)
                else:
                    arr_pad_row = np.full(max_seq_len - x.shape[1], tokenizer.pad_token_id)

                for i in range(batch_size):
                    arr_pad_row_col[i] = np.concatenate([x[i], arr_pad_row])
                    #import pdb; pdb.set_trace()
                lst_pad_mc_generated_tokens.append(arr_pad_row_col)
                #import pdb; pdb.set_trace()
        
        # tokenizer.batch_decode(lst_mc_generated_tokens[0])
        arr_mc_generated_tokens = np.array(lst_pad_mc_generated_tokens) # (mc_drop_num, batch size, seq_len)
        # import pdb; pdb.set_trace()
        return arr_mc_generated_tokens


    # Majority Voting
    def run_majority_vote(tokenizer, arr_mc_generated_tokens, args):
        logger.info("***** Running Majority Vote *****")

        mc_drop_num = args.mc_drop_num
        batch_size = len(arr_mc_generated_tokens[0])
        max_seq_len = len(arr_mc_generated_tokens[0][0])

        arr_max_vote_pred = np.full((batch_size, max_seq_len), -100)
        arr_num_max_vote_pred = np.full((batch_size, 1), -100)

        for i in range(batch_size):
            ith_batch_votes = arr_mc_generated_tokens[:, i, :]
            lst_ith_batch_votes = ith_batch_votes.tolist()
             # vote table
            votes_table = {}

            for vote in lst_ith_batch_votes:
                tuple_vote = tuple(vote)
                # check if key in table
                if tuple_vote in votes_table:  
                    # increment counter  
                    votes_table[tuple_vote] += 1 
                else:
                    # create counter for vote
                    votes_table[tuple_vote] = 1  
            # find max pred
            max_vote_pred = max(votes_table, key=votes_table.get)
            arr_max_vote_pred[i] = max_vote_pred
            # find max pred's vote
            num_max_vote_pred = votes_table[max_vote_pred]
            arr_num_max_vote_pred[i] = num_max_vote_pred

        #import pdb; pdb.set_trace()
        pred_label = torch.tensor(arr_max_vote_pred)#.to(device)
        num_vote_pred_label = torch.tensor(arr_num_max_vote_pred)#.to(device)
        
        # proportion of the best vote num 
        num_vote_pred_label = num_vote_pred_label / args.mc_drop_num

        if args.ignore_pad_token_for_loss:
            arr_max_vote_pred = np.where(arr_max_vote_pred != -100, arr_max_vote_pred, tokenizer.pad_token_id)

        logger.info('===================================')
        logger.info('Max votes preds : ')
        logger.info(tokenizer.batch_decode(arr_max_vote_pred, skip_special_tokens=True))
        logger.info('Max votes num: ')
        logger.info( arr_num_max_vote_pred.tolist())
        logger.info('Max votes num / mc_drop_num: ')
        logger.info( num_vote_pred_label.tolist())
        

        return pred_label, num_vote_pred_label


    def test_time_tuning_soft(args, model, tokenizer, test_time_tuning_dataloader, lst_batch_with_pred_labels):

        for epoch in range(0, args.test_time_tuning_epoch):
            model.train()
            total_loss = 0
            for step, batch in enumerate(test_time_tuning_dataloader):
                batch['decoder_input_ids'] = None
                batch['labels'] = lst_batch_with_pred_labels[step][0] # dummy labels for matching shape
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    lm_logits = outputs.logits  # SHAPE: (BATCH, MAX_SEQ_LENGTH, VOCAB SIZE)
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    all_pred_loss = 0
                    for pred_labels in lst_batch_with_pred_labels[step]:
                        each_pred_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), pred_labels.view(-1))
                        all_pred_loss = all_pred_loss + each_pred_loss
                        # print(each_pred_loss)
                    # print(all_pred_loss)
                    all_pred_loss = all_pred_loss / args.mc_drop_num
                    # print(all_pred_loss)

                    # loss = outputs.loss
                    accelerator.backward(all_pred_loss)
                    optimizer.step()
                    lr_scheduler_test_time_tuning.step()
                    optimizer.zero_grad() 

                    # We keep track of the loss at each epoch
                    total_loss = total_loss + all_pred_loss.cpu().detach().float()

            logger.info("Epoch %d Loss:{} ".format(total_loss / len(lst_batch_with_pred_labels)), epoch) 
            #import pdb; pdb.set_trace()

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

    def test_time_tuning(args, model, tokenizer, filtered_test_time_tuning_dataloader):
        for epoch in range(0, args.test_time_tuning_epoch):
            model.train()
            total_loss = 0
            for step, batch in enumerate(filtered_test_time_tuning_dataloader):
                batch['decoder_input_ids'] = None
                # import pdb; pdb.set_trace()
                # tokenizer.decode(batch['input_ids'][0])
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler_test_time_tuning.step()
                    optimizer.zero_grad() 

                    # We keep track of the loss at each epoch
                    total_loss = total_loss + loss.cpu().detach().float()

            logger.info("Epoch %d Loss:{} ".format(total_loss / len(filtered_test_time_tuning_dataloader)), epoch) 
            #import pdb; pdb.set_trace()

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



    # Validation
    if args.do_eval:
        logger.info("***** Running Validation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        if args.val_max_answer_length is None:
            args.val_max_answer_length = args.max_answer_length

        gen_kwargs = {
            "max_length": args.val_max_answer_length,
            #'no_repeat_ngram_size':2
            #"num_beams": args.num_beams,
        }

        # test time tuning
        model.eval()
        if args.do_test_time_tuning:
            lst_batch_with_best_pred_labels = []
            lst_batch_with_all_pred_labels = []
            lst_all_gold_labels = []

            lst_filtered_input_ids = []
            lst_filtered_labels = []
            filtered_test_time_tuning_examples = {}

            for step, batch in enumerate(test_time_tuning_dataloader):
                gold_labels = batch['labels'].cpu()
                # output
                lst_all_gold_labels.append([gold_labels])
                
                # mc drop
                lst_mc_generated_tokens = run_mc_drop(model, tokenizer, batch, gen_kwargs, args)

                # make array with MC drop results
                # (mc_drop_num, batch size, seq_len)
                arr_mc_generated_tokens = convert_to_arr(lst_mc_generated_tokens, args)
                #import pdb; pdb.set_trace()

                # prepare soft_label batches
                lst_mc_preds_batch = []
                for arr_mc_generated_token in arr_mc_generated_tokens:
                    if args.do_soft_label:
                        all_pred_label = torch.tensor(arr_mc_generated_token).to(device)
                    else:
                        all_pred_label = torch.tensor(arr_mc_generated_token)
                    lst_mc_preds_batch.append(all_pred_label)
                    
                lst_batch_with_all_pred_labels.append(lst_mc_preds_batch)
                # import pdb; pdb.set_trace()

                # majority vote for the best pred & its vote num
                best_pred_label, proportion_best_pred_label = run_majority_vote(tokenizer, arr_mc_generated_tokens, args)
                

                batch_size = len(batch['input_ids'])
                for filtered_idx in range(batch_size):
                    if proportion_best_pred_label[filtered_idx] >= args.filter_thres:
                        #import pdb; pdb.set_trace()
                        lst_filtered_input_ids.append(batch['input_ids'][filtered_idx])
                        lst_filtered_labels.append(best_pred_label[filtered_idx])

                # TODO 나중에 지우기..
                # update batch labels to the best predicted labels
                best_pred_batch = copy.deepcopy(batch)
                best_pred_batch['labels'] = best_pred_label
                best_pred_batch['proportion_best_pred_labels'] = proportion_best_pred_label
                lst_batch_with_best_pred_labels.append(best_pred_batch)
                            
            assert len(lst_filtered_input_ids) == len(lst_filtered_labels)
            filtered_test_time_tuning_examples['input_ids'] = lst_filtered_input_ids
            filtered_test_time_tuning_examples['labels'] = lst_filtered_labels
            #import pdb; pdb.set_trace()

            filtered_test_time_tuning_examples = datasets.Dataset.from_dict(filtered_test_time_tuning_examples)

            filtered_test_time_tuning_dataloader = DataLoader(
            filtered_test_time_tuning_examples, collate_fn=data_collator, shuffle=False, batch_size=args.per_device_eval_batch_size
            )

            filtered_test_time_tuning_dataloader = accelerator.prepare(
                filtered_test_time_tuning_dataloader
            )

            # prepare max_train_steps, train_epoch, lr_scheduler
            args.max_test_time_train_steps, args.test_time_tuning_epoch, lr_scheduler_test_time_tuning = prepare_scheduler(args, accelerator, filtered_test_time_tuning_dataloader, optimizer, args.max_test_time_train_steps, args.test_time_tuning_epoch)


            # TODO 나중에 지우기..
            # save pickle
            with open(os.path.join(args.output_dir, "lst_batch_with_all_pred_labels.pickle"), 'wb') as out_pkl_file:
                pickle.dump(lst_batch_with_all_pred_labels, out_pkl_file)
            with open(os.path.join(args.output_dir, "gold_labels.pickle"), 'wb') as out_pkl_file:
                pickle.dump(lst_all_gold_labels, out_pkl_file)
             

            # test_time_tuning
            # soft label
            if args.do_soft_label:
                logger.info("***** Running Test-time tuning *****")
                logger.info(f"  Num examples = {len(test_time_tuning_dataset)}")

                #import pdb; pdb.set_trace()
                args.max_test_time_train_steps, args.test_time_tuning_epoch, lr_scheduler_test_time_tuning = prepare_scheduler(args, accelerator, test_time_tuning_dataloader, optimizer, args.max_test_time_train_steps, args.test_time_tuning_epoch)

                test_time_tuning_soft(args, model, tokenizer, test_time_tuning_dataloader, lst_batch_with_all_pred_labels)
                del lst_batch_with_all_pred_labels
                del test_time_tuning_dataloader
            # hard label
            else:
                logger.info("***** Running Test-time tuning *****")
                logger.info(f"  Num examples = {len(filtered_test_time_tuning_examples)}")
                test_time_tuning(args, model, tokenizer, filtered_test_time_tuning_dataloader)
                del filtered_test_time_tuning_dataloader
                del filtered_test_time_tuning_examples

        # actual inference
        model.eval()
        all_gen_tokens = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():

                generated_tokens = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                all_gen_tokens.append(generated_tokens)

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                logger.info('==========================================')
                logger.info(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))
                logger.info('Prediction : ')
                logger.info(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
                logger.info('Answer : ')
                logger.info(tokenizer.batch_decode(labels, skip_special_tokens=True))


        max_len = max([x.shape[1] for x in all_gen_tokens])  # Get the max_length of the tensor
        # concatenate the numpy array
        gen_tokens_concat = create_and_fill_np_array(all_gen_tokens, eval_dataset, max_len)

        # delete the list of numpy arrays
        del all_gen_tokens
        prediction = post_processing_function(tokenizer, args, raw_datasets, eval_examples, eval_dataset, gen_tokens_concat)
        #import pdb; pdb.set_trace()


        prediction_label_ids = prediction.label_ids
        prediction_predictions = prediction.predictions

        assert len(prediction_label_ids) == len(prediction_predictions)

        final_em_score, final_f1_score = calculate_f1_em(prediction_label_ids, prediction_predictions)


        final_eval_results = {'final_em_score' : final_em_score, 'final_f1_score': final_f1_score}

        logger.info(f"Evaluation metrics: {final_eval_results}")
        print(final_eval_results)

        with open(os.path.join(args.output_dir, "final_eval_results.json"), "w") as f:
            json.dump(final_eval_results, f)

        #import pdb; pdb.set_trace()

        if args.dataset_name is not None:
            # TODO) 나중에 지우기
            eval_metric = metric.compute(predictions=prediction_predictions, references=prediction_label_ids)
            logger.info(f"Evaluation metrics: {eval_metric}")

            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)

                    eval_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
                        json.dump(eval_results, f)



if __name__ == "__main__":
    main()





