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
                #import pdb; pdb.set_trace()

                dict_final['answers'] = {'text' : first_selected_answer + data['answers']}
                dict_final['context'] = lst_context[c_idx].lower()
                dict_final['id'] = str(example_id)
                dict_final['question'] = data['paragraphs']['qas'][0]['question'].lower()
                dict_final['title'] =  data['title'].lower()
                dict_final['c_idx'] = c_idx
                dict_final['a_idx'] = a_idx

                lst_dict_final.append(dict_final)
                #import pdb; pdb.set_trace()


                return

lst_dict_final = []
example_id = 0

with open('./data/cpgqa/original/dpr/cpgQA-v1.0.json', 'r') as input_file:
    json_data = json.load(input_file)
    
    print(len(json_data['data']))

    for data in json_data['data']:
        data['answers'] = [data['paragraphs']['qas'][0]['answers'][0]['text'].lower()]
        lst_answer = [a for a in data['answers']]
        lst_context = [data['paragraphs']['context'].lower()]
        # data['positive_ctxs'][0]['text'] = data['positive_ctxs'][0]['text'].lower()
        
        
        run_nested_loop(lst_context, lst_answer, example_id, lst_dict_final)

        example_id = example_id + 1

        #import pdb; pdb.set_trace()
        
with open("./data/cpgqa/preprocessed/cpgqa_test.json", "w") as output_file:
    json.dump(lst_dict_final, output_file, indent=4, sort_keys=True)


print(len(lst_dict_final))