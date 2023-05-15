import torch
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# ALPHABET TO TOKEN MAPPING
mapper = {
    'A': tokenizer('A').input_ids[0],
    'B': tokenizer('B').input_ids[0],
    'C': tokenizer('C').input_ids[0],
    'D': tokenizer('D').input_ids[0],
    'EOS': tokenizer('A').input_ids[1]
}


# BATCH SIZE: 2
input_ids = tokenizer(
    [
        "Select one among A, B, C, and D.",
        "Select one among A, B, C, and D."
    ], 
    return_tensors="pt"
).input_ids
labels = tokenizer(
    [
        "A",
        "A"
    ], 
    return_tensors="pt"
).input_ids

# A: 0.6, B: 0.1, C: 0.1, D: 0.2
# 10번의 예측에서 A가 6번, D가 2번, B와 C가 1번씩 나옴
pred_labels = torch.Tensor([
    [6, 1, 1, 2],
    [6, 1, 1, 2]
])
pred_labels = pred_labels / 10  # Convert Values to Probabilities

outputs = model(input_ids=input_ids, labels=labels)
lm_logits = outputs.logits  # SHAPE: (BATCH, LABEL_TOKENS, VOCAB SIZE)

real_targets = torch.zeros_like(lm_logits)
for target, label in zip(real_targets, pred_labels):
    assert len(target) == 2

    target[0, [mapper['A'], mapper['B'], mapper['C'], mapper['D']]] = label
    target[1, mapper['EOS']] = 1

loss_fct = CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), real_targets.view(-1, real_targets.size(-1)))


import pdb; pdb.set_trace()