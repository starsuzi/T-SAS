import torch
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
import datasets
import numpy as np
import math

from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint


# TODO: peft
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import PeftModel, PeftConfig


def load_model(args):
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
        
    return model, tokenizer


def preprocess_dataset(args, raw_datasets):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    #column_names = raw_datasets["train"].column_names
    column_names = raw_datasets[args.val_column].column_names
    
    # Get the column names for input/target.
    question_column = args.question_column
    if question_column not in column_names:
        raise ValueError(
            f"--question_column' value '{args.question_column}' needs to be one of: {', '.join(column_names)}"
        )

    context_column = args.context_column
    if context_column not in column_names:
        raise ValueError(
            f"--context_column' value '{args.context_column}' needs to be one of: {', '.join(column_names)}"
        )


    answer_column = args.answer_column
    if answer_column not in column_names:
        raise ValueError(
            f"--answer_column' value '{args.answer_column}' needs to be one of: {', '.join(column_names)}"
        )

    return question_column, context_column, answer_column


def preprocess_features_function(examples, args, raw_datasets, tokenizer):
    question_column, context_column, answer_column = preprocess_dataset(args, raw_datasets)

    # Temporarily set max_answer_length for training.
    max_answer_length = args.max_answer_length
    padding = "max_length" if args.pad_to_max_length else False
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)



    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    
    
    # ("Read this and answer the question\n\n{context}\n\n{question}", "{answer}"),
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "left"#"right"
    examples[context_column] =  ['Read this and answer the question\n\n{}'.format(c.strip()) for c in examples[context_column]]
    examples[question_column] = ['\n\n{}'.format(q.strip()) for q in examples[question_column]]
 

    # Please answer a question about the following article. Question: {}\n Article:'
    # Padding side determines if we do (question|context) or (context|question).
    # pad_on_right = tokenizer.padding_side == "right"
    # examples[question_column] =  ['Please answer a question about the following article. Question: {}\n Article:'.format(q.lstrip()) for q in examples[question_column]] #[q.lstrip() for q in examples[question_column]]


    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    model_inputs = tokenizer(
        examples[question_column if pad_on_right else context_column],
        examples[context_column if pad_on_right else question_column],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )   

    if args.dataset_name == 'sciq':
        targets = examples[answer_column]
    elif args.dataset_name == 'covid_qa_deepset':
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in examples[answer_column]]
    else:
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in examples[answer_column]]

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    model_inputs["example_id"] = []
    # Augment the overflowing tokens to the labels
    labels_out = []

    #import pdb; pdb.set_trace()

    for i in range(len(model_inputs["input_ids"])):
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        model_inputs["example_id"].append(examples["id"][sample_index])
        labels_out.append(labels["input_ids"][sample_index])

    model_inputs["labels"] = labels_out
    # tokenizer.batch_decode(model_inputs["input_ids"])[0]
    return model_inputs


# Post-processing:
def post_processing_function(
    tokenizer, args, raw_datasets, examples: datasets.Dataset, features: datasets.Dataset, outputs, stage="eval"
):
    # Decode the predicted tokens.
    preds = outputs
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    #labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #import pdb; pdb.set_trace()
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
    predictions = {}
    # Let's loop over all the examples!
    #import pdb; pdb.set_trace()
    for example_index, example in enumerate(examples):
        # This is the index of the feature associated to the current example.
        #import pdb; pdb.set_trace()
        feature_index = feature_per_example[example_index]
        predictions[example["id"]] = decoded_preds[feature_index]
        #import pdb; pdb.set_trace()

    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k if type(k) == str else str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k if type(k) == str else str(k), "prediction_text": v} for k, v in predictions.items()]

    if args.dataset_name == 'sciq':
        _, _, answer_column = preprocess_dataset(args, raw_datasets)
        references = [{"id": ex["id"], "answers": {'text': [ex[answer_column]], 'answer_start': []}} for ex in examples]
    else:
        _, _, answer_column = preprocess_dataset(args, raw_datasets)
        references = [{"id": ex["id"] if type(ex["id"]) == str else str(ex["id"]), "answers": ex[answer_column] if ex[answer_column]['text'] != [] else {'text': [''], 'answer_start': []}} for ex in examples]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)



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



def prepare_scheduler(args, accelerator, dataloader, optimizer, max_train_steps, train_epoch):
    overrode_max_train_steps = False

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if max_train_steps is None:
        max_train_steps = train_epoch * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if overrode_max_train_steps:
        max_train_steps = train_epoch * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    train_epoch = math.ceil(max_train_steps / num_update_steps_per_epoch)

    return max_train_steps, train_epoch, lr_scheduler

# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""
    
    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example - 
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]
        
    return gold_answers

def calculate_f1_em(prediction_label_ids, prediction_predictions):
    # report f1 and em
    total_f1_score = 0
    total_em_score = 0
    for (gold_ans, pred) in zip(prediction_label_ids, prediction_predictions):
        gold_answers = gold_ans['answers']['text']
        prediction = pred['prediction_text']
        # if len(gold_answers) > 1:
        #     import pdb; pdb.set_trace()

        f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
        em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)

        total_f1_score = total_f1_score + f1_score
        total_em_score = total_em_score + em_score

    final_em_score = (total_em_score / len(prediction_label_ids)) * 100
    final_f1_score = (total_f1_score / len(prediction_label_ids)) * 100

    return final_em_score, final_f1_score