import json


# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

with open('./outputs/squad_dpr/context/gpt3/squad_dpr_dev_gpt3_2023_08_27_T05_07_52.json', 'r') as input_file:
    json_data = json.load(input_file)
    print(len(json_data))

    total_em_score = 0
    total_f1_score = 0
    
    for data in json_data:
        gold_answers = data['answer']
        prediction = data['prediction']

        f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
        em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)

        total_f1_score = total_f1_score + f1_score
        total_em_score = total_em_score + em_score

    final_f1_score = (total_f1_score / len(json_data)) * 100
    final_em_score = (total_em_score / len(json_data)) * 100

    print('em')
    print(final_em_score)
    print('f1')
    print(final_f1_score)


    #import pdb; pdb.set_trace()
