import json
import datasets
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

{
    "answers": {
        "answer_start": [1],
        "text": ["This is a test text"]
    },
    "context": "This is a test context.",
    "id": "1",
    "question": "Is this a test?",
    "title": "train test"
}

lst_dict_final = []
example_id = 0
total_answer_len = 0
total_answer_num = 0
max_answer_len = 0

with open('/data/syjeong/prompt_test/data/nq/original/dpr/biencoder-nq-dev.json', 'r') as input_file:
    json_data = json.load(input_file)
    print(len(json_data))

    for data in json_data:
        dict_final = {}

        # preprocess answer text
        data['answers'][0] = data['answers'][0].replace(' ,', ',')
        data['answers'][0] = data['answers'][0].replace(' - ', '-')
        data['answers'][0] = data['answers'][0].replace(" 's", "'s")
        data['answers'][0] = data['answers'][0].replace('( ', '(')
        data['answers'][0] = data['answers'][0].replace(' )', ')')
        data['answers'][0] = data['answers'][0].replace(' / ', '/')
        data['answers'][0] = data['answers'][0].replace(' .', '.')
        data['answers'][0] = data['answers'][0].replace(' %', '%')
        data['answers'][0] = data['answers'][0].replace("s '", "s'")
        
        #import pdb; pdb.set_trace()
        data['answers'] = [i.lower() for i in data['answers']]
        data['positive_ctxs'][0]['text'] = data['positive_ctxs'][0]['text'].lower()

        # if the context does not contain the answer
        if data['answers'][0] not in data['positive_ctxs'][0]['text']:
            continue
        assert data['answers'][0] in data['positive_ctxs'][0]['text']

        #import pdb; pdb.set_trace()

        for answer in data['answers']:
            total_answer_num = total_answer_num + 1
            total_answer_len = total_answer_len + len(tokenizer(answer)['input_ids'])
            if max_answer_len < len(tokenizer(answer)['input_ids']):
                max_answer_len = len(tokenizer(answer)['input_ids'])

        example_id = example_id + 1
        dict_final['answers'] = {'text' : data['answers']}
        dict_final['context'] = data['positive_ctxs'][0]['text']
        dict_final['id'] = str(example_id)
        dict_final['question'] = data['question']
        dict_final['title'] = data['positive_ctxs'][0]['title']

        #import pdb; pdb.set_trace()

        lst_dict_final.append(dict_final)

        
# with open("/data/syjeong/prompt_test/data/nq/preprocessed/nq_dev.json", "w") as output_file:
#     json.dump(lst_dict_final, output_file, indent=4, sort_keys=True)

print(total_answer_len / total_answer_num)
print(len(lst_dict_final))
print(max_answer_len)
import pdb; pdb.set_trace()
