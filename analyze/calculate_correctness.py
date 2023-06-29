import os
import numpy as np
from tqdm import tqdm
import pickle, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

lst_gold_labels_path = './outputs/nq/context/baseline/topk/model/google/flan-t5-large/filter_thres/-1/orig_prompt/no_lora/topk_num/15/temperature/0.5/epoch/5/2023_06_21/01_01_36/gold_labels.pickle'
lst_batch_with_all_pred_labels_path = './outputs/nq/context/baseline/topk/model/google/flan-t5-large/filter_thres/-1/orig_prompt/no_lora/topk_num/15/temperature/0.5/epoch/5/2023_06_21/01_01_36/lst_batch_with_all_pred_labels.pickle'

# load
with open(lst_gold_labels_path, 'rb') as f_lst_gold_labels_path:
    lst_gold_labels = pickle.load(f_lst_gold_labels_path)
# load
with open(lst_batch_with_all_pred_labels_path, 'rb') as f_lst_batch_with_all_pred_labels_path:
    lst_batch_with_all_pred_labels = pickle.load(f_lst_batch_with_all_pred_labels_path)

assert len(lst_gold_labels) == len(lst_batch_with_all_pred_labels)

lst_final_confident_pred = []
lst_final_gold_answer = []

total_freq_pred_value_proportion = 0
total_divsersity = 0

for gold_allPred in zip(lst_gold_labels, lst_batch_with_all_pred_labels):
    mc_drop_gold_labels = gold_allPred[0]
    mc_drop_all_pred_labels = gold_allPred[1]

    batch_size = len(mc_drop_all_pred_labels[0])
    
    for jth_batch in range(batch_size): 
        lst_mc_preds = []
        lst_mc_preds_tok = []
        
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
            
            pad_indices = np.where(ith_mc_drop_jth_sample_in_batch == 0)
            ith_mc_drop_jth_sample_in_batch_without_special_tok = np.delete(ith_mc_drop_jth_sample_in_batch, pad_indices)
            eos_indices = np.where(ith_mc_drop_jth_sample_in_batch_without_special_tok == 1)
            ith_mc_drop_jth_sample_in_batch_without_special_tok = np.delete(ith_mc_drop_jth_sample_in_batch_without_special_tok, eos_indices)
            lst_mc_preds_tok = lst_mc_preds_tok + ith_mc_drop_jth_sample_in_batch_without_special_tok.tolist() 


        #print('==============')
        #print(lst_mc_preds)
        import pdb; pdb.set_trace()
        
        dict_mc_freq_preds = Counter(lst_mc_preds)
        freq_pred = max(dict_mc_freq_preds, key=dict_mc_freq_preds.get)
        lst_final_confident_pred.append(freq_pred)
        
        freq_pred_value = dict_mc_freq_preds[freq_pred]
        
        # freq vote threshold 
        freq_pred_value_proportion = freq_pred_value / mc_drop_num
        # total
        total_freq_pred_value_proportion = total_freq_pred_value_proportion + freq_pred_value_proportion

        #lexicl diversity
        divsersity = len(set(lst_mc_preds_tok)) / len(lst_mc_preds_tok)
        total_divsersity = total_divsersity + divsersity



#total_freq_pred_value_proportion / len(lst_gold_labels)
#total_divsersity / len(lst_gold_labels)
#import pdb; pdb.set_trace()
print('===topk===')
print('lexical diversity')
print(total_divsersity / len(lst_final_confident_pred))
print('max vote / mc_drop_num')
print(total_freq_pred_value_proportion / len(lst_final_confident_pred))
#import pdb; pdb.set_trace()
