import json
from tqdm import tqdm

with open('./outputs/nq/context/gpt3/nq_dev_gpt3_2023_08_26_T10_11_36.json', 'r') as input_file:
    json_data = json.load(input_file)
    print(len(json_data))

    for idx, data in tqdm(enumerate(json_data)):
        data['context_question_text'] = data['context_question_text'][idx]

with open("./outputs/nq/context/gpt3/nq_dev_gpt3.json", "w") as output_file:
    json.dump(json_data, output_file, indent=4, sort_keys=True)

    #import pdb; pdb.set_trace()