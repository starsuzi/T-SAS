import json

def run_nested_loop(lst_context, lst_answer, example_id, lst_dict_final) :
    dict_final = {}
    for c_idx, ctx in enumerate(lst_context):
        for a_idx, ans in enumerate(lst_answer):
            if ans in ctx:
                dict_final['answers'] = {'text' : data['answers']}
                dict_final['context'] = data['positive_ctxs'][c_idx]['text'].lower()
                dict_final['id'] = str(example_id)
                dict_final['question'] = data['question'].lower()
                dict_final['title'] = data['positive_ctxs'][c_idx]['title'].lower()
                dict_final['c_idx'] = c_idx
                dict_final['a_idx'] = a_idx

                lst_dict_final.append(dict_final)

                return

lst_dict_final = []
example_id = 0

with open('./data/trivia/original/biencoder-trivia-dev.json', 'r') as input_file:
    json_data = json.load(input_file)
    print(len(json_data))

    for data in json_data:
        if data['positive_ctxs'] == []:
            continue

        data['answers'] = [i.lower() for i in data['answers']]
        lst_answer = [a for a in data['answers']]
        lst_context = [c['text'].lower() for c in data['positive_ctxs']]        
        
        run_nested_loop(lst_context, lst_answer, example_id, lst_dict_final)

        example_id = example_id + 1
        
with open("./data/trivia/preprocessed/trivia_dev.json", "w") as output_file:
    json.dump(lst_dict_final, output_file, indent=4, sort_keys=True)


print(len(lst_dict_final))