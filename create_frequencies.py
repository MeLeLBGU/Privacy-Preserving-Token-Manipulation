import argparse
import sys

# import pandas as pd
import torch.nn as nn
import train as training
from transformers import AutoTokenizer, BertForSequenceClassification
from datasets import load_dataset
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from BertClassifier import BertClassifier
import train_new as train2
import test as predict
from utils import *
import pickle

def read_and_generate_frequencies(tokenizer, data):
    freq_ids = []
    vocab = create_vocabulary(tokenizer=tokenizer)

    freq_ids_map = {int(key): 0 for key in vocab}
    for sent in tqdm(data):
        input_ids = tokenizer.encode(sent)
        for val in input_ids:
            val = int(val)
            if val in special:
                continue
            freq_ids_map[val] = freq_ids_map[val] + 1
    log.info("Done generating input ids from text!")

    return freq_ids_map

def check_tokens2(tokenizer1, tokenizer2, data):
    i = 0
    print("starting to check tokens")
    for sent in tqdm(data):
        i = i + 1
        input1 = tokenizer1.encode(sent)
        input2 = tokenizer2.encode(sent)
        if len(input1) - 2 != len(input2):
            print("Tokenizers are not the same:", sent)
            print("Roberta:", input1)
            print("GPT-2:  ", input2)
            exit(1)
        for word in sent:
            input1 = tokenizer1.encode(word)
            input2 = tokenizer2.encode(word)
            if len(input1) - 2 != len(input2):
                print("Tokenizers are not the same:", sent)
                print("Roberta:", input1)
                print("GPT-2:  ", input2)
                exit(1)
        if i%1000 == 0:
            print("Sentence ", sent)
            print("Roberta:", len(input1))
            print("GPT-2:  ", len(input2))

def add_to_remapper(tkz1, tkz2):
    a_file = open("roberta_gpt_mapper.pkl_save", "rb")
    tokenizer_mapper = pickle.load(a_file)
    a_file.close()
    d = tkz1.get_vocab()
    v1 = [0] * len(d)
    for i, key in enumerate(d):
        v1[i] = d[key]
    bad_tokens = []
    for i, t1 in enumerate(v1):
        if tokenizer_mapper[t1] == -1:
            skip = False
            s1 = tkz1.decode(t1)
            t2 = tkz2.encode(s1)
            if len(t2) > 1:
                print("Bad token ,we skip it", t1, t2, s1)
                bad_tokens.append([t1,t2,s1])
                continue
            t2 = t2[0]
            # check no doubles
            for i, t in enumerate(tokenizer_mapper):
                if tokenizer_mapper[t] == t2:# check if we didnt add the t2 value already
                    print("Double booking:", t1, t2, s1)
                    bad_tokens.append([t1,t2,s1])
                    skip = True
                    break
            if skip:
                continue
            tokenizer_mapper[t1] = t2
    a_file = open("roberta_gpt_mapper.pkl", "wb")
    pickle.dump(tokenizer_mapper, a_file)
    a_file.close()
    print(len(bad_tokens))

def compare_tokenizers(tkz1, tkz2, data):
    #tkz1 is roberta
    bad_tokens = []
    d = tkz1.get_vocab()
    v1 = [0] * len(d)
    for i, key in enumerate(d):
        v1[i] = d[key]
    tokenizer_mapper = {int(key): -1 for key in v1}
    # for i, t1 in enumerate(v1):
    #     s1 = tkz1.decode(t1)
    #     # try:
    #     #     s1.encode(encoding='utf-8').decode('ascii')
    #     # except UnicodeDecodeError:
    #     #     bad_tokens.append(t1)
    #     #     continue


    #     t2 = tkz2.encode(s1)
    #     if t1 in [23133, 18164, 42393, 21402]:
    #         print(t1, s1, t2, tkz2.decode(t2))
    #     if len(t2) > 1:
    #         bad_tokens.append(t1)
    #     else:
    #         if tokenizer_mapper[t1] != -1:
    #             print("two mapped tokens")
    #             exit(1)
    #         tokenizer_mapper[t1] = t2[0]
    # a_file = open("roberta_gpt_mapper.pkl", "wb")
    # pickle.dump(tokenizer_mapper, a_file)
    # a_file.close()
    print("bad tokens", bad_tokens)
    print("bad tokens", len(bad_tokens))
    skip = 0
    i = 0
    for sent in data:
        i = i + 1
        roberta_tkns = tkz1.encode(sent, add_special_tokens = False, pad_to_max_length = False)
        gpt_real_tkns = tkz2.encode(sent, add_special_tokens = False, pad_to_max_length = False)
        if len(roberta_tkns) != len(gpt_real_tkns):
            print("not same length papi")
            exit(1)
        for j, tkn in enumerate(roberta_tkns):
            if tokenizer_mapper[tkn] != -1 and tokenizer_mapper[tkn] != gpt_real_tkns[j]:
                print("Two tokens in gpt go to 1 token in roberta")
                exit(1)
            tokenizer_mapper[tkn] = gpt_real_tkns[j]

        gpt_tkns = [tokenizer_mapper[t] for t in roberta_tkns]
        if tkz2.decode(gpt_tkns) != tkz2.decode(gpt_real_tkns) or len(gpt_real_tkns) != len(gpt_tkns):
            print("Sentence", sent)
            print("ROBERTA    :", roberta_tkns)
            print("ROBERTA-GPT:", gpt_tkns)
            print("GPT        :", gpt_real_tkns)
            print("StringsRB:", tkz1.decode(roberta_tkns))
            print("StringsRG:", tkz2.decode(gpt_tkns))
            print("StringsGP:", tkz2.decode(gpt_real_tkns))
            print("skipped:", skip)
            exit(1)

        if i % 10000 == 0:
            print("Step:", i)
            print("skipper:", skip)
            print("Sentence", sent)
            print("ROBERTA    :", roberta_tkns)
            print("ROBERTA-GPT:", gpt_tkns)
            print("GPT        :", gpt_real_tkns)
            print("StringsRB:", tkz1.decode(roberta_tkns))
            print("StringsRG:", tkz2.decode(gpt_tkns))
            print("StringsGP:", tkz2.decode(gpt_real_tkns))
            zeros = 0       
            for tkn in tokenizer_mapper:
                if tokenizer_mapper[tkn] == -1:
                    zeros = zeros + 1
            print("zeros", zeros)

    a_file = open("roberta_gpt_mapper.pkl", "wb")
    pickle.dump(tokenizer_mapper, a_file)
    a_file.close()
        # if any(x in roberta_tkns for x in bad_tokens):
        #     skip = skip + 1
        #     continue
        # gpt_tkns = [tokenizer_mapper[t] for t in roberta_tkns]
        # if tkz2.decode(gpt_tkns) != tkz2.decode(gpt_real_tkns) or len(gpt_real_tkns) != len(gpt_tkns):
        #     print("Sentence", sent)
        #     print("ROBERTA    :", roberta_tkns)
        #     print("ROBERTA-GPT:", gpt_tkns)
        #     print("GPT        :", gpt_real_tkns)
        #     print("StringsRB:", tkz1.decode(roberta_tkns))
        #     print("StringsRG:", tkz2.decode(gpt_tkns))
        #     print("StringsGP:", tkz2.decode(gpt_real_tkns))
        #     print("skipped:", skip)
        #     exit(1)
#49223, 50083, 50151, 50192, 46774, 49333, 3, 49469, 50264, 49624, 41613, 0, 50261, 50263, 50079, 50223, 34955, 1, 2, 49272, 50211, 49957, 49739, 50032, 50186, 48943, 29462, 50262
# this is from roberta---gpt
#2311, 73, 13090, 645, 569, 18354, 7496, 513, 1058, 791, 47398, 17740, 357, 4960, 1058, 20543, 4210, 4210, 4210, 4210, 4210, 5641, 44444, 9202, 25084, 24440, 12675, 11839, 18,  11, 6578,  13, 569, 18354, 7496, 286, 262, 30193, 513, 1267,  11, 8811, 6412, 284, 355, 569, 18354, 7496, 17740, 6711, 2354, 2869, 11 , 318, 257, 16106, 2597, 2488, 12, 31, 2712, 2008, 983, 4166, 416, 29490, 290, 6343, 13, 44206, 329, 262, 14047, 44685,  13, 28728, 287, 3269, 2813, 287, 2869, 11 , 340, 318, 262, 2368, 983, 287, 262, 569, 18354, 7496, 2168, 13 , 12645, 278, 262, 976, 21748, 286, 16106, 290, 1103, 2488, 12, 31, 640, 11327, 355, 663, 27677, 11 , 262, 1621, 4539, 10730, 284, 262, 717, 983, 290, 5679, 262, 366, 17871, 5321, 366, 11 , 257, 23634, 2422, 4326, 7351, 262, 3277, 286, 7096, 544, 1141, 262, 5498, 1898, 6839, 1810, 508, 1620, 3200, 2042, 4560, 290, 389, 46852, 1028, 262, 11773, 4326, 366, 2199, 321, 265, 88, 12552, 366, 13 , 220, 198
# this is from gpt
#2311, 73, 13090, 645, 569, 18354, 7496, 513, 1058, 791, 47398, 17740, 357, 4960, 1058, 10545, 230 , 99  , 161 , 254 , 112 , 5641, 44444, 9202, 25084, 24440, 12675, 11839, 18, 837, 6578, 764, 569, 18354, 7496, 286, 262, 30193, 513, 1267, 837, 8811, 6412, 284, 355, 569, 18354, 7496, 17740, 6711, 2354, 2869, 837, 318, 257, 16106, 2597, 2488, 12, 31, 2712, 2008, 983, 4166, 416, 29490, 290, 6343, 13, 44206, 329, 262, 14047, 44685, 764, 28728, 287, 3269, 2813, 287, 2869, 837, 340, 318, 262, 2368, 983, 287, 262, 569, 18354, 7496, 2168, 764, 12645, 278, 262, 976, 21748, 286, 16106, 290, 1103, 2488, 12, 31, 640, 11327, 355, 663, 27677, 837, 262, 1621, 4539, 10730, 284, 262, 717, 983, 290, 5679, 262, 366, 17871, 5321, 366, 837, 257, 23634, 2422, 4326, 7351, 262, 3277, 286, 7096, 544, 1141, 262, 5498, 1898, 6839, 1810, 508, 1620, 3200, 2042, 4560, 290, 389, 46852, 1028, 262, 11773, 4326, 366, 2199, 321, 265, 88, 12552, 366, 764, 220, 198   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default = "result.pt", type = str,
                        dest = 'save', help = 'Save mode path.')
    parser.add_argument('--corpus', default = "wikitext-103-raw-v1", type = str,
                        dest = 'corpus', help = 'What corpus to use for tokenizer')
    parser.add_argument('--model', default = "bert-base-uncased", type = str,
                        dest = 'model', help = 'What base model to use')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model,errors="strict")
    tokenizer2 = AutoTokenizer.from_pretrained("gpt2",errors="strict")

    add_to_remapper(tokenizer, tokenizer2)
    exit(1)
    if "wiki" in args.corpus:
        corpus = load_dataset("wikitext", args.corpus)
        text = corpus["train"]["text"] + corpus["test"]["text"] + corpus["validation"]["text"]
    else:
        text = load_dataset(args.corpus)
    compare_tokenizers(tokenizer, tokenizer2, text)
    # check_tokens2(tokenizer, tokenizer2, text)
    print("FINISHED GOOD!")
    exit(1)
    data = read_and_generate_frequencies(tokenizer, text)
    a_file = open(args.save, "wb")
    pickle.dump(data, a_file)
    a_file.close()