from collections import defaultdict
import string
import tokenizers
import re
from transformers import BertTokenizer
import random


def create_vocabulary(data={}, tokenizer=BertTokenizer, vocab_type="token"):
    if vocab_type == "token":
        d = tokenizer.get_vocab()
        arr = [0] * len(d)
        for i, key in enumerate(d):
            arr[i] = d[key]
        return arr

    if vocab_type == "word":
        exit(1)
        word_freqs = defaultdict(int)
        pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
        for i, sent in enumerate(data):
            words_with_offsets = pre_tokenizer.pre_tokenize_str(sent)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                if word in string.punctuation:
                    continue
                word_freqs[str(word)] += 1
        return word_freqs


def shuffle_vocab(vocab: list):
    random.shuffle(vocab)
    return vocab


def remap_vocab(vocab: list, remap_type=2):
    word_map = vocab
    for i, token in enumerate(vocab):
        if i % remap_type == 0:
            new_token = token
            word_map[i] = token  # just to make sure
        else:
            word_map[i] = new_token
    return word_map


# we assume that the vocab is id: 'word'
def remap_input_ids(input_ids, vocab):
    new_input_ids = input_ids
    for i, ids in enumerate(input_ids):
        for j, token in enumerate(ids):
            if token == 0:
                continue
            new_input_ids[i][j] = vocab[token]
    return new_input_ids

