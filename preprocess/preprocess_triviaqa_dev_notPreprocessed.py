import json
import datasets

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

with open('/data/soyeong/prompt_test/data/trivia/original/dpr/biencoder-trivia-dev.json', 'r') as input_file:
    json_data = json.load(input_file)
    print(len(json_data))

    for data in json_data:
        if data['positive_ctxs'] == []:
            continue

        dict_final = {}

        # preprocess answer text
        # data['answers'][0] = data['answers'][0].replace(' ,', ',')
        # data['answers'][0] = data['answers'][0].replace(' - ', '-')
        # data['answers'][0] = data['answers'][0].replace(" 's", "'s")
        # data['answers'][0] = data['answers'][0].replace('( ', '(')
        # data['answers'][0] = data['answers'][0].replace(' )', ')')
        # data['answers'][0] = data['answers'][0].replace(' / ', '/')
        # data['answers'][0] = data['answers'][0].replace(' .', '.')
        # data['answers'][0] = data['answers'][0].replace(' %', '%')
        # data['answers'][0] = data['answers'][0].replace("s '", "s'")
        
        # data['answers'] = [i.lower() for i in data['answers']]
        # data['positive_ctxs'][0]['text'] = data['positive_ctxs'][0]['text'].lower()

        # if the context does not contain the answer
        # data['answers'][0] in data['positive_ctxs'][1]['text']
        # if data['answers'][0] not in data['positive_ctxs'][0]['text']:
        #     import pdb; pdb.set_trace()
        #     continue
        # assert data['answers'][0] in data['positive_ctxs'][0]['text']

        example_id = example_id + 1
        dict_final['answers'] = {'text' : data['answers']}
        dict_final['context'] = data['positive_ctxs'][0]['text']
        dict_final['id'] = str(example_id)
        dict_final['question'] = data['question']
        dict_final['title'] = data['positive_ctxs'][0]['title']

        #import pdb; pdb.set_trace()

        lst_dict_final.append(dict_final)

        
with open("/data/soyeong/prompt_test/data/trivia/preprocessed/trivia_dev_all.json", "w") as output_file:
    json.dump(lst_dict_final, output_file, indent=4, sort_keys=True)

print(len(lst_dict_final))