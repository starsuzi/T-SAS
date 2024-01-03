import json
import datasets

# {
#     "answers": {
#         "answer_start": [1],
#         "text": ["This is a test text"]
#     },
#     "context": "This is a test context.",
#     "id": "1",
#     "question": "Is this a test?",
#     "title": "train test"
# }

def run_nested_loop(lst_context, lst_answer, example_id, lst_dict_final) :
    dict_final = {}
    for c_idx, ctx in enumerate(lst_context):
        for a_idx, ans in enumerate(lst_answer):
            if ans in ctx:
                first_selected_answer =[data['answers'][a_idx]]

                dict_final['answers'] = {'text' :  data['answers']}
                dict_final['context'] = data['ctxs'][c_idx]['text'].lower()
                dict_final['id'] = str(example_id)
                dict_final['question'] = data['question'].lower()
                dict_final['title'] = data['ctxs'][c_idx]['title'].lower()
                dict_final['c_idx'] = c_idx
                dict_final['a_idx'] = a_idx

                lst_dict_final.append(dict_final)
                #import pdb; pdb.set_trace()


                return

lst_dict_final = []
example_id = 0

with open('./data/sleepqa/original/sleep-test.json', 'r') as input_file:
    json_data = json.load(input_file)
    print(len(json_data))

    for data in json_data:
        if data['ctxs'] == []:
            continue

        

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
        
        data['answers'] = [i.lower() for i in data['answers']]
        lst_answer = [a for a in data['answers']]
        lst_context = [c['text'].lower() for c in data['ctxs']]
        # data['positive_ctxs'][0]['text'] = data['positive_ctxs'][0]['text'].lower()
        
        
        run_nested_loop(lst_context, lst_answer, example_id, lst_dict_final)

        example_id = example_id + 1

        #import pdb; pdb.set_trace()
        
with open("./data/sleepqa/preprocessed/sleepqa_test.json", "w") as output_file:
    json.dump(lst_dict_final, output_file, indent=4, sort_keys=True)


print(len(lst_dict_final))