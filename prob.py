import torch
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

input_ids = tokenizer('Answer the following question by reasoning step-by-step. The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?', padding=True, return_tensors="pt").input_ids

#outputs = model.generate(input_ids, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
outputs = model.generate(input_ids, return_dict_in_generate=True, output_scores=True)
transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)

output_length = input_ids.shape[1] + np.sum(transition_scores.numpy() < 0, axis=1)
probabilities = torch.exp(transition_scores.sum(axis=1) / (output_length))

import pdb; pdb.set_trace()
tokenizer.batch_decode(outputs.sequences)
# https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075?page=2

# input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# encoder-decoder models, like BART or T5.
# input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
# generated_tokens = outputs.sequences[:, input_length:]
