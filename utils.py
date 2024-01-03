import torch
import logging

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

logger = logging.getLogger(__name__)


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
    pad_on_right = tokenizer.padding_side == "left"
    examples[context_column] =  ['Read this and answer the question\n\n{}'.format(c.strip()) for c in examples[context_column]]
    examples[question_column] = ['\n\n{}'.format(q.strip()) for q in examples[question_column]]


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

    for i in range(len(model_inputs["input_ids"])):
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        model_inputs["example_id"].append(examples["id"][sample_index])
        labels_out.append(labels["input_ids"][sample_index])

    model_inputs["labels"] = labels_out

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
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
    predictions = {}
    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # This is the index of the feature associated to the current example.
        feature_index = feature_per_example[example_index]
        predictions[example["id"]] = decoded_preds[feature_index]

    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k if type(k) == str else str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k if type(k) == str else str(k), "prediction_text": v} for k, v in predictions.items()]

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


# MC-drop
def run_mc_drop(model, accelerator, tokenizer, batch, gen_kwargs, args):
    logger.info("***** Running MC Drop *****")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    lst_generated_tokens = []
    with torch.no_grad():
        for i in range(0, args.mc_drop_num):
            model.train()

            outputs = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # generated_tokens
            generated_tokens = outputs.sequences
            generated_tokens = accelerator.gather_for_metrics(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()

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
            logger.info('Answer : ')
            logger.info(tokenizer.batch_decode(gold_labels, skip_special_tokens=True))

            # delete tokenizer.pad_token_id
            generated_tokens = np.array([[(t if t != tokenizer.pad_token_id else -100) for t in tok] for tok in generated_tokens])
            
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            lst_generated_tokens.append(generated_tokens)

    # make array with MC drop results
    # (mc_drop_num, batch size, seq_len)
    arr_mc_generated_tokens = convert_to_arr(lst_generated_tokens, args)

    return arr_mc_generated_tokens


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
            lst_pad_mc_generated_tokens.append(arr_pad_row_col)
    
    arr_mc_generated_tokens = np.array(lst_pad_mc_generated_tokens) # (mc_drop_num, batch size, seq_len)
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

    pred_label = torch.tensor(arr_max_vote_pred)
    num_vote_pred_label = torch.tensor(arr_num_max_vote_pred)
    
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


def test_time_tuning(args, model, accelerator, optimizer, lr_scheduler_test_time_tuning, tokenizer, filtered_test_time_tuning_dataloader):
    for epoch in range(0, args.test_time_tuning_epoch):
        model.train()
        total_loss = 0
        for step, batch in enumerate(filtered_test_time_tuning_dataloader):
            batch['decoder_input_ids'] = None
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

        f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
        em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)

        total_f1_score = total_f1_score + f1_score
        total_em_score = total_em_score + em_score

    final_em_score = (total_em_score / len(prediction_label_ids)) * 100
    final_f1_score = (total_f1_score / len(prediction_label_ids)) * 100

    return final_em_score, final_f1_score