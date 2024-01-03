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

## Run
We run our proposed T-SAS models as follows:
```bash
# XL size
$ bash ./run/nq/xl/tsas/run_nq_tsas.sh
$ bash ./run/squad/xl/tsas/run_squad_tsas.sh
$ bash ./run/trivia/xl/tsas/run_trivia_tsas.sh

# Large size
$ bash ./run/nq/large/naive_baseline/run_nq_baseline_large.sh
$ bash ./run/squad/large/naive_baseline/run_squad_baseline_large.sh
$ bash ./run/trivia/large/naive_baseline/run_trivia_baseline_large.sh
```
Note that for the models larger than 3B, we trained
them adopting a low-rank adaptation (LoRA)
method.

For the naïve baseline models, run as follows:
```bash
# XL size
$ bash ./run/nq/xl/naive_baseline/run_nq_baseline.sh
$ bash ./run/squad/xl/naive_baseline/run_squad_baseline.sh
$ bash ./run/trivia/xl/naive_baseline/run_trivia_baseline.sh

# Large size
$ bash ./run/nq/large/naive_baseline/run_nq_baseline_large.sh
$ bash ./run/squad/large/naive_baseline/run_squad_baseline_large.sh
$ bash ./run/trivia/large/naive_baseline/run_trivia_baseline_large.sh
```
## Citation
If you found the provided code with our paper useful, we kindly request that you cite our work.
```BibTex
@inproceedings{DBLP:conf/emnlp/JeongBCHP23,
  author       = {Soyeong Jeong and
                  Jinheon Baek and
                  Sukmin Cho and
                  Sung Ju Hwang and
                  Jong Park},
  title        = {Test-Time Self-Adaptive Small Language Models for Question Answering},
  booktitle    = {Findings of the Association for Computational Linguistics: {EMNLP}
                  2023, Singapore, December 6-10, 2023},
  pages        = {15459--15469},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.findings-emnlp.1033},
  biburl       = {https://dblp.org/rec/conf/emnlp/JeongBCHP23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```