import json
import jsonlines
import csv
from tqdm import tqdm

# {
#     "answers": {
#         "text": ["This is a test text", 'another']
#     },
#     "context": "This is a test context.",
#     "id": "1",
#     "question": "Is this a test?",
#     "title": "train test"
# }



dict_total_qa = {}
with jsonlines.open('/data/syjeong/prompt_test/data/hotpotqa/original/beir/queries.jsonl') as qa_file:
    for qa_line in qa_file.iter():
        dict_total_qa[qa_line['_id']] = qa_line
        #import pdb; pdb.set_trace()

lst_dict_test = []
with open("/data/syjeong/prompt_test/data/hotpotqa/original/beir/qrels/test.tsv") as test_tsv_file:
    test_tsv_file = csv.reader(test_tsv_file, delimiter="\t")
    for idx, test_line in enumerate(test_tsv_file):
        if idx != 0:
            dict_test = {}
            qid = test_line[0]
            cid = test_line[1]
            dict_test = {'qid': qid, 'cid': cid}
            lst_dict_test.append(dict_test)
            #import pdb; pdb.set_trace()
            #dict_test[test_line[0]] = {'qid':test_line[0], 'cid':test_line[1]}


dict_total_context = {}
with jsonlines.open('/data/syjeong/prompt_test/data/hotpotqa/original/beir/corpus.jsonl') as context_file:
    for context_line in tqdm(context_file.iter()):
        dict_total_context[context_line['_id']] = context_line

lst_dict_final = []
lst_existing_qid = []
for idx, dict_test in enumerate(lst_dict_test):
    dict_final = {}
    qid = dict_test['qid']
    cid = dict_test['cid']

    # dict_total_qa[qid]
    # dict_total_context[cid]
    # try:
    dict_final['answers'] = {'text' : [dict_total_qa[qid]['metadata']['answer']]}
    dict_final['context'] =  dict_total_context[cid]['text']
    dict_final['id'] = str(idx)
    dict_final['question'] =  dict_total_qa[qid]['text']
    dict_final['title'] = dict_total_context[cid]['title']
    dict_final['qid'] = qid
    dict_final['cid'] = cid
    lst_dict_final.append(dict_final)
    # except:
    #     print(qid, cid)
    #     continue


with open("/data/syjeong/prompt_test/data/hotpotqa/preprocessed/hotpotqa_test.json", "w") as output_file:
    json.dump(lst_dict_final, output_file, indent=4, sort_keys=True)
