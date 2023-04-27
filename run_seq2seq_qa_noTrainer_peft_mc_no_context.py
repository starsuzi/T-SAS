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

question_answering_column_name_mapping = {
    "squad_v2": ("question", "answer"),
    #"squad" : ("question", "context", "answer"),
    "covid_qa_deepset": ("question", "answer"),
    "pubmed_qa": ("question", "long_answer"),
}


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
    parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
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
    
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
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

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
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
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.



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
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""


    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    #import pdb; pdb.set_trace()

    # Get the column names for input/target.
    dataset_columns = question_answering_column_name_mapping.get(args.dataset_name, None)
    
    if args.question_column is None:
        question_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        question_column = args.question_column
        if question_column not in column_names:
            raise ValueError(
                f"--question_column' value '{args.question_column}' needs to be one of: {', '.join(column_names)}"
            )


    if args.answer_column is None:
        answer_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        answer_column = args.answer_column
        if answer_column not in column_names:
            raise ValueError(
                f"--answer_column' value '{args.answer_column}' needs to be one of: {', '.join(column_names)}"
            )

    if args.val_max_answer_length is None:
        args.val_max_answer_length = args.max_answer_length
        
    # Temporarily set max_answer_length for training.
    max_answer_length = args.max_answer_length
    padding = "max_length" if args.pad_to_max_length else False

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_squad_batch(
        examples,
        question_column: str,
        answer_column: str,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        answers = examples[answer_column]

        def generate_input(_question):
            return " ".join(["question:", _question.lstrip()])

        inputs = [generate_input(question) for question in questions]
        #import pdb; pdb.set_trace()
        if args.dataset_name == 'sciq':
            targets = answers
        else:
            targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
            
        return inputs, targets

    def preprocess_function(examples):
        inputs, targets = preprocess_squad_batch(examples, question_column, answer_column)

        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Tokenize targets with text_target=...
        labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # test tuning preprocessing
    def preprocess_test_time_tuning_function(examples):
        inputs, targets = preprocess_squad_batch(examples, question_column, answer_column)

        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Tokenize targets with text_target=...
        labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Validation preprocessing
    def preprocess_validation_function(examples):
        inputs, targets = preprocess_squad_batch(examples, question_column, answer_column)

        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Tokenize targets with text_target=...
        labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if args.train_column not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets[args.train_column]
    if args.max_train_samples is not None:
        # We will select sample from whole data if agument is specified
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # Create train feature from dataset
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        if args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(args.max_train_samples))
    
    
    #import pdb; pdb.set_trace()
    if args.val_column not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_examples = raw_datasets[args.val_column]
    if args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(args.max_eval_samples))
    # Validation Feature Creation
    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(
            preprocess_validation_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    if args.max_eval_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))


    if args.do_test_time_tuning:
        if args.val_column not in raw_datasets:
            raise ValueError("--do_test_time_tuning requires a eval dataset")
        test_time_tuning_examples = raw_datasets[args.val_column]
        if args.max_test_time_tuning_samples is not None:
            # We will select sample from whole data
            test_time_tuning_examples = test_time_tuning_examples.select(range(args.max_test_time_tuning_samples))
        # test_time_tuning Feature Creation
        with accelerator.main_process_first():
            test_time_tuning_dataset = test_time_tuning_examples.map(
                preprocess_test_time_tuning_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on eval dataset",
            )
        if args.max_test_time_tuning_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            test_time_tuning_dataset = test_time_tuning_dataset.select(range(args.max_test_time_tuning_samples))



    if args.do_predict:
        if args.test_column not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets[args.test_column]
        if args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(args.max_predict_samples))
        # Predict Feature Creation
        with accelerator.main_process_first():
            predict_dataset = predict_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(args.max_predict_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )


    # Post-processing:
    def post_processing_function(
        examples: datasets.Dataset, features: datasets.Dataset, outputs, stage="eval"
    ):
        # Decode the predicted tokens.
        #import pdb; pdb.set_trace()
        preds = outputs
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        #labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        #import pdb; pdb.set_trace()

        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in enumerate(decoded_preds)
            ]
        else:
            formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in enumerate(decoded_preds)]

        #import pdb; pdb.set_trace()
        references = [{"id": str(ex_id), "answers": {'text': [ex[answer_column]], 'answer_start': []}} for ex_id, ex in enumerate(examples)]
        #references = [{"id": ex_id, "answers": ex[answer_column] if ex[answer_column]['text'] != [] else {'text': [''], 'answer_start': []}} for ex_id, ex in enumerate(examples)]

        #import pdb; pdb.set_trace()
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    if args.do_test_time_tuning:
        #import pdb; pdb.set_trace()
        test_time_tuning_dataloader = DataLoader(
            test_time_tuning_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )        

    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(
            predict_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
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

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.do_test_time_tuning:
        model, optimizer, train_dataloader, eval_dataloader, test_time_tuning_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, test_time_tuning_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

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
    #import pdb; pdb.set_trace()
    metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(all_gen_tokens, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
        Args:
            all_gen_tokens(:obj:`tensor`):
                This is the output predictions of the model.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """
        
        step = 0
        # create a numpy array and fill it with -100.
        gen_toks_concat = np.full((len(dataset), max_len), -100)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, gen_tok in enumerate(all_gen_tokens):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step
            #import pdb; pdb.set_trace()
            batch_size = gen_tok.shape[0]
            cols = gen_tok.shape[1]

            if step + batch_size < len(dataset):
                gen_toks_concat[step : step + batch_size, :cols] = gen_tok
            else:
                gen_toks_concat[step:, :cols] = gen_tok[: len(dataset) - step]

            step += batch_size

        return gen_toks_concat

    # Train!
    if args.do_train:
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
            if args.with_tracking:
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
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    logger.info("Loss:{} ".format(loss))

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

                generated_tokens = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                #import pdb; pdb.set_trace()
                generated_tokens = accelerator.gather_for_metrics(generated_tokens)
                generated_tokens = generated_tokens.cpu().numpy()
                # delete bos token
                generated_tokens = generated_tokens[:, 1:] 

                logger.info('==========================================')
                logger.info(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))
                logger.info('Prediction : ')
                logger.info(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
                # logger.info('Answer : ')
                # logger.info(tokenizer.batch_decode(labels, skip_special_tokens=True))


                # delete tokenizer.pad_token_id
                generated_tokens = np.array([[(t if t != tokenizer.pad_token_id else -100) for t in tok] for tok in generated_tokens])
                #import pdb; pdb.set_trace()
                

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                lst_generated_tokens.append(generated_tokens)


        #import pdb; pdb.set_trace()
        return lst_generated_tokens

    # Majority Voting
    def run_majority_vote(tokenizer, lst_mc_generated_tokens, args):
        logger.info("***** Running Majority Vote *****")

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
        # TODO: lst_mc_generated_tokens padding 필요..
        arr_mc_generated_tokens = np.array(lst_pad_mc_generated_tokens) # (mc_drop_num, batch size, seq_len)
        arr_max_vote_pred = np.full((batch_size, max_seq_len), -100)

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

        #import pdb; pdb.set_trace()
        pred_label = torch.tensor(arr_max_vote_pred).to(device)

        if args.ignore_pad_token_for_loss:
            arr_max_vote_pred = np.where(arr_max_vote_pred != -100, arr_max_vote_pred, tokenizer.pad_token_id)

        logger.info('===================================')
        logger.info('Max votes preds : ')
        logger.info(tokenizer.batch_decode(arr_max_vote_pred, skip_special_tokens=True))

        return pred_label


    def test_time_tuning(model, tokenizer, orig_batch, pred_label, args):
        logger.info("***** Running Test-time tuning *****")

        logger.info('Answer : ')
        #import pdb; pdb.set_trace()
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            logger.info(tokenizer.batch_decode(np.where(orig_batch['labels'].cpu() != -100, orig_batch['labels'].cpu(), tokenizer.pad_token_id), skip_special_tokens=True))
        else:
            logger.info(tokenizer.batch_decode(orig_batch['labels'], skip_special_tokens=True))

        batch = copy.deepcopy(orig_batch)
        #import pdb; pdb.set_trace()
        batch['labels'] = pred_label
        batch['decoder_input_ids'] = None

        for epoch in range(0, args.test_time_tuning_epoch):
            model.train()
            #import pdb; pdb.set_trace()
            with accelerator.accumulate(model):
                #import pdb; pdb.set_trace()
                #torch.set_grad_enabled(True)
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

        del batch 


    # Validation
    if args.do_eval:
        logger.info("***** Running Validation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        gen_kwargs = {
            "max_length": args.val_max_answer_length,
            #'no_repeat_ngram_size':2
            #"num_beams": args.num_beams,
        }

        # test time tuning
        model.eval()
        if args.do_test_time_tuning:
            for step, batch in enumerate(test_time_tuning_dataloader):
                #import pdb; pdb.set_trace()
                lst_mc_generated_tokens = run_mc_drop(model, tokenizer, batch, gen_kwargs, args)
                #import pdb; pdb.set_trace()
                pred_label = run_majority_vote(tokenizer, lst_mc_generated_tokens, args)
                test_time_tuning(model, tokenizer, batch, pred_label, args)

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

        prediction = post_processing_function(eval_examples, eval_dataset, gen_tokens_concat)
        #import pdb; pdb.set_trace()
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f"Evaluation metrics: {eval_metric}")

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

                eval_results = {f"eval_{k}": v for k, v in eval_metric.items()}
                with open(os.path.join(args.output_dir, "valid_results.json"), "w") as f:
                    json.dump(eval_results, f)

    # Prediction
    if args.do_predict:
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(predict_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        all_gen_tokens = []

        model.eval()

        gen_kwargs = {
            "max_length": args.val_max_answer_length,
            #"num_beams": args.num_beams,
        }

        for step, batch in enumerate(predict_dataloader):
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
                logger.info(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)[0])
                logger.info('Prediction : ' + str(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]))
                logger.info('Answer : ' + str(tokenizer.batch_decode(labels, skip_special_tokens=True)[0]))
                #decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                #decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        max_len = max([x.shape[1] for x in all_gen_tokens])  # Get the max_length of the tensor
        # concatenate the numpy array
        gen_tokens_concat = create_and_fill_np_array(all_gen_tokens, predict_dataset, max_len)

        # delete the list of numpy arrays
        del all_gen_tokens

        prediction = post_processing_function(predict_examples, predict_dataset, gen_tokens_concat)
        predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f"Predict metrics: {predict_metric}")

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

                predict_results = {f"test_{k}": v for k, v in predict_metric.items()}
                with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
                    json.dump(predict_results, f)


if __name__ == "__main__":
    main()





