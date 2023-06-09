import pickle
import torch
import numpy as np
from collections import Counter

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

lst_gold_labels_path = '/data/syjeong/prompt_test/outputs/nq/context/test_time_tuning/model/google/flan-t5-xl/orig_prompt/lora/filter_thres/mc/20/2023_06_09/01_40_09/gold_labels.pickle'
lst_batch_with_all_pred_labels_path = '/data/syjeong/prompt_test/outputs/nq/context/test_time_tuning/model/google/flan-t5-xl/orig_prompt/lora/filter_thres/mc/20/2023_06_09/01_40_09/lst_batch_with_all_pred_labels.pickle'


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
        #import pdb; pdb.set_trace()
        gold_answers = gold_ans
        prediction = pred
        # if len(gold_answers) > 1:
        #     import pdb; pdb.set_trace()

        f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
        em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)

        total_f1_score = total_f1_score + f1_score
        total_em_score = total_em_score + em_score

    final_em_score = (total_em_score / len(prediction_label_ids)) * 100
    final_f1_score = (total_f1_score / len(prediction_label_ids)) * 100

    return final_em_score, final_f1_score

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl") 


# load
with open(lst_gold_labels_path, 'rb') as f_lst_gold_labels_path:
    lst_gold_labels = pickle.load(f_lst_gold_labels_path)
# load
with open(lst_batch_with_all_pred_labels_path, 'rb') as f_lst_batch_with_all_pred_labels_path:
    lst_batch_with_all_pred_labels = pickle.load(f_lst_batch_with_all_pred_labels_path)

assert len(lst_gold_labels) == len(lst_batch_with_all_pred_labels)

lst_final_confident_pred = []
lst_final_gold_answer = []

for gold_allPred in zip(lst_gold_labels, lst_batch_with_all_pred_labels):
    mc_drop_gold_labels = gold_allPred[0]
    mc_drop_all_pred_labels = gold_allPred[1]

    batch_size = len(mc_drop_all_pred_labels[0])
    
    for jth_batch in range(batch_size): 
        lst_mc_preds = []
        # gold answer
        #import pdb; pdb.set_trace()
        gold_answer = np.where(mc_drop_gold_labels[0][jth_batch] != -100, mc_drop_gold_labels[0][jth_batch], tokenizer.pad_token_id)
        gold_answer_text = tokenizer.decode(gold_answer, skip_special_tokens=True)
        lst_final_gold_answer.append([gold_answer_text])
        mc_drop_num = len(mc_drop_all_pred_labels)
        for ith_mc_drop in range(mc_drop_num):
            #import pdb; pdb.set_trace()
            mc_drop_all_pred_labels[ith_mc_drop][jth_batch]

            ith_mc_drop_jth_sample_in_batch = np.where(mc_drop_all_pred_labels[ith_mc_drop][jth_batch] != -100, mc_drop_all_pred_labels[ith_mc_drop][jth_batch], tokenizer.pad_token_id)
            ith_mc_drop_jth_sample_in_batch_text = tokenizer.decode(ith_mc_drop_jth_sample_in_batch, skip_special_tokens=True)
            lst_mc_preds.append(ith_mc_drop_jth_sample_in_batch_text)
        #print('==============')
        #print(lst_mc_preds)
        
        dict_mc_freq_preds = Counter(lst_mc_preds)
        freq_pred = max(dict_mc_freq_preds, key=dict_mc_freq_preds.get)
        lst_final_confident_pred.append(freq_pred)
        
        freq_pred_value = dict_mc_freq_preds[freq_pred]
        
        # freq vote threshold 
        freq_pred_value_proportion = freq_pred_value / mc_drop_num

        # variance threshold
        lst_vote_values = [i for i in dict_mc_freq_preds.values()]
        lst_vote_values_filled = lst_vote_values + [0] * (mc_drop_num - len(lst_vote_values))
        
        variance = np.var(lst_vote_values_filled)


        # mc_drop_all_pred_labels[0]: 0번쨰 mcdrop의 배치
        # mc_drop_all_pred_labels[0][1] ~ mc_drop_all_pred_labels[4][1] 랑 비교해야함. 
        # 0번째 mcdrop의 배치내 1번째 샘플,  4번째 mcdrop의 배치내 1번째 샘플

        if freq_pred_value_proportion < 0.5:
            #import pdb; pdb.set_trace()
            lst_final_gold_answer = lst_final_gold_answer[:-1]
            lst_final_confident_pred = lst_final_confident_pred[:-1]
            


final_em_score, final_f1_score = calculate_f1_em(lst_final_gold_answer, lst_final_confident_pred)
#
print('em:')
print(final_em_score)
print('f1')
print(final_f1_score)        
print('total_len')
print(len(lst_final_gold_answer))

#import pdb; pdb.set_trace()
    
        



