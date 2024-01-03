# Test-Time Self-Adaptive Small Language Models for Question Answering

Official Code Repository for the paper "Test-Time Self-Adaptive Small Language Models for Question Answering" (Findings of EMNLP 2023): https://aclanthology.org/2023.findings-emnlp.1033.pdf

## Abstract
Recent instruction-finetuned large language models (LMs) have achieved notable performances in various tasks, such as questionanswering (QA). However, despite their ability to memorize a vast amount of general knowledge across diverse tasks, they might be suboptimal on specific tasks due to their limited capacity to transfer and adapt knowledge to target tasks. Moreover, further finetuning LMs with labeled datasets is often infeasible due to their absence, but it is also questionable if we can transfer smaller LMs having limited knowledge only with unlabeled test data. In this work, we show and investigate the capabilities of smaller self-adaptive LMs, only with unlabeled test data. In particular, we first stochastically generate multiple answers, and then ensemble them while filtering out lowquality samples to mitigate noise from inaccurate labels. Our proposed self-adaption strategy demonstrates significant performance improvements on benchmark QA datasets with higher robustness across diverse prompts, enabling LMs to stay stable.

## Installation
```bash
$ conda create -n tsas python=3.8
$ conda activate tsas
$ pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install -r requirements.txt
```

## Dataset
We download Natural Question, TriviaQA, and SQuAD from https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py.
For example, we download Natural Question as follows:
```bash
$ mkdir -p data/nq/original
$ mkdir -p data/nq/preprocessed
$ cd data/nq/original
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
$ gzip -d biencoder-nq-dev.json.gz
```
Then, we should preprocess datasets as follows:
```bash
$ python preprocess/preprocess_nq_dev.py
$ python preprocess/preprocess_trivia_dev.py
$ python preprocess/preprocess_squad_dev.py
```


