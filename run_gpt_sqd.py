import argparse
import json
import logging
import pickle
from datasets import load_dataset
import openai
from time import time
from datetime import datetime
from tqdm import tqdm

current_timestamp = time()
str_time = datetime.fromtimestamp(current_timestamp).strftime('%Y_%m_%d_T%H_%M_%S')

def askGPT(input_text):
    openai.api_key = "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"
    response = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = input_text,
        max_tokens = 30,
    )
    print(response.choices[0].text)
    return response.choices[0].text

data_files = {}

validation_file = 'data/squad_dpr/preprocessed/squad_dpr_dev.json'
data_files["validation"] = validation_file
extension = validation_file.split(".")[-1]
#import pdb; pdb.set_trace()
raw_datasets = load_dataset(extension, data_files=data_files)

eval_examples = raw_datasets['validation']

context_question_text = ['Passage:{} // Question: {} // Referring to the passage above, the correct answer to the given question is // Answer:'.format(c.strip(), q.strip()) for c, q in zip(eval_examples['context'], eval_examples['question'])]

# temp
#context_question_text = context_question_text[:5]

lst_dict_response = []
for idx, input_text in tqdm(enumerate(context_question_text)):
    response = askGPT(input_text)
    dict_response = {'id': eval_examples['id'][idx], 'context_question_text': context_question_text[idx], 'answer': eval_examples['answers'][idx]['text'], 'prediction':response}
    lst_dict_response.append(dict_response)

with open("./outputs/squad_dpr/context/gpt3/squad_dpr_dev_gpt3_"+str_time+".json", "w") as output_file:
    json.dump(lst_dict_response, output_file, indent=4, sort_keys=True)

print(len(lst_dict_response))
#import pdb; pdb.set_trace()
