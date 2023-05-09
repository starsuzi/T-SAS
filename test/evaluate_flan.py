import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    #     (Pdb) df.iloc[idx]
    # 0    Find the degree for the given field extension ...
    # 1                                                    0
    # 2                                                    4
    # 3                                                    2
    # 4                                                    6
    # 5                                                    B
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    # import pdb; pdb.set_trace()
    # 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:'
    return prompt


def gen_prompt(train_df, subject, k=-1):
    # (Pdb) train_df
    #                                                 0           1             2            3            4  5
    # 0  Find all c in Z_3 such that Z_3[x]/(x^2 + c) i...           0             1            2            3  B
    # 1  Statement 1 | If aH is an element of a factor ...  True, True  False, False  True, False  False, True  B
    # 2  Statement 1 | Every element of a group generat...  True, True  False, False  True, False  False, True  C
    # 3  Statement 1| Every function from a finite set ...  True, True  False, False  True, False  False, True  A
    # 4            Find the characteristic of the ring 2Z.           0             3           12           30  A

    # (Pdb) prompt
    # 'The following are multiple choice questions (with answers) about  abstract algebra.\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nStatement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: B\n\nStatement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: C\n\nStatement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: A\n\nFind the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: A\n\n'
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    #import pdb; pdb.set_trace()
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    # rows with long prompt
    lst_long_row = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        # if subject == 'high_school_european_history':
        #     import pdb; pdb.set_trace()

        
        # if tokenizer(prompt_end, return_tensors="pt").input_ids.shape[-1] > 384:
        #     #import pdb; pdb.set_trace()
        #     lst_long_row.append(i)
        #     continue

        while input_ids.shape[-1] > 384 and k >=0: #2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        if k < 0 :
            lst_long_row.append(i)
            continue
            #import pdb; pdb.set_trace

        label = test_df.iloc[i, test_df.shape[1] - 1]

        decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        decoder_input_ids = model._shift_right(decoder_input_ids)
        logits = model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids
        ).logits.flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        #import pdb; pdb.set_trace()
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)



    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    # Drop rows with long prompts
    test_df = test_df.drop(labels=lst_long_row, axis=0)

    return cors, acc, all_probs, test_df


def main(args):

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    heads_per_gpu = len(model.encoder.block) // args.ngpu
    device_map = {
        gpu: list(
            range(
                0 + (gpu * heads_per_gpu),
                (0 + (gpu * heads_per_gpu)) + heads_per_gpu,
            )
        )
        for gpu in range(args.ngpu)
    }
    model.parallelize(device_map)
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs, test_df = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=1)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="google/flan-t5-xl",
    )
    args = parser.parse_args()
    main(args)
