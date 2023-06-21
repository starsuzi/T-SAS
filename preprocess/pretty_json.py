import json

with open('/data/syjeong/prompt_test/data/hotpotqa/original/dpr/hotpot_train_v1.1.json') as json_file:
    json_data = json.load(json_file)

with open('/data/syjeong/prompt_test/data/hotpotqa/original/dpr/hotpot_train_v1.1_pretty.json', "w") as writer: 
    writer.write(json.dumps(json_data, indent=4) + "\n")
