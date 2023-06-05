import argparse
import sys

# import pandas as pd
import torch.nn as nn
import train as training
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from BertClassifier import BertClassifier
import train_new as train2
import test as predict
from utils import *
import text_manipulation
import naive_attacker
import pickle


def read_and_generate_frequencies(tokenizer, data):
    freq_ids = []
    vocab = text_manipulation.create_vocabulary(tokenizer=tokenizer)

    freq_ids_map = {int(key): 0 for key in vocab}
    for sent in tqdm(data):
        input_ids = tokenizer.encode(sent)
        for i, ids in enumerate(input_ids):
            for val in ids:
                val = int(val)
                if val == 0:
                    continue
                freq_ids_map[val] = freq_ids_map[val] + 1
    log.info("Done generating input ids from text!")

    return freq_ids_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default = "result.pt", type = str,
                        dest = 'save', help = 'Save mode path.')
    parser.add_argument('--corpus', default = "wikitext-103-raw-v1", type = str,
                        dest = 'corpus', help = 'What corpus to use for tokenizer')
    parser.add_argument('--model', default = "bert-base-uncased", type = str,
                        dest = 'model', help = 'What base model to use')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model)

    if "wiki" in args.corpus:
        corpus = load_dataset("wikitext", args.corpus)
        text = corpus["train"]["text"] + corpus["test"]["text"] + corpus["validation"]["text"]
    else:
        text = load_dataset(args.corpus)
    data = read_and_generate_frequencies(tokenizer, text)
    a_file = open(save, "wb")
    pickle.dump(dictionary_data, a_file)
    a_file.close()