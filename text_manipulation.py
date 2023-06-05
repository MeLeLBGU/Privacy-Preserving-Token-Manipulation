from collections import defaultdict
import string
import tokenizers
import re
from transformers import BertTokenizer
import random
import utils
import collections
import logging as log
import tqdm

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




def mapper(remap_type, vocab:list, input_ids):
    if remap_type == "random":
        log.info("You chose to remap the inputs ids randomly")
        return remap_vocab_randomly(vocab)
    elif remap_type == "freq-high":
        log.info("You chose to remap the inputs ids by mapping the lower frequency to the higher one")
        return remap_vocab_by_frequency(vocab, input_ids, freq_type="high")
    elif remap_type == "freq-low":
        log.info("You chose to remap the inputs ids by mapping the higher frequency to the lower one")
        return remap_vocab_by_frequency(vocab, input_ids, freq_type="low")


def remap_vocab_randomly(vocab: list, remap_count=2, shuffle=True):
    """
    given a vocab of the form [id]:id.
     The main idea is to remap such that [1]:1 and [2]:1 --- token with id
     1 and 2 will be remapped to 1.
     So first we get the indices we want to remap, which is in this case [2]
     and if we want to shuffle (for example [1]:1 and [5]:1) we will shuffle the
     indices.
     afterwards we will remap accordingly
    """
    save_vocab = vocab.copy()
    reverse_map = {}
    lst = []
    new_vocab = vocab
    indices_to_shuffle = [i for i in range(len(vocab)) if (i % remap_count != 0)]

    if shuffle:
        random.shuffle(indices_to_shuffle)
    k = 0
    for i, token in enumerate(vocab):
        if i % remap_count == 0:
            lst = []
            new_token = token
            lst.append(i) # for the reverse vocabulary
        else:
            new_vocab[indices_to_shuffle[k]] = new_token
            lst.append(indices_to_shuffle[k])
            k = k + 1
        if (i + 1) % remap_count == 0:
            reverse_map[int(new_token)] = lst

    return new_vocab, reverse_map


def remap_vocab_by_frequency(vocab: list, input_ids, freq_type="high"):
    freq_ids_map = {int(key): 0 for key in vocab}
    # freq_ids = collections.Counter() didnt work, crashed for some reason
    for i, ids in enumerate(input_ids):
        for val in ids:
            val = int(val)
            if val == 0:
                break
            freq_ids_map[val] = freq_ids_map[val] + 1
    sorted_freq_ids = sorted(freq_ids_map.items(), key = lambda x: x[1])
    remap = {int(key): 0 for key in vocab}
    reverse_map = {int(key): None for key in vocab}

    for i, tpl in tqdm(enumerate(sorted_freq_ids)):

        high_freq_id, _ = sorted_freq_ids[-(i + 1)]
        low_freq_id, _ = sorted_freq_ids[i]
        if high_freq_id in utils.UNIQUE_TOKENS or low_freq_id in utils.UNIQUE_TOKENS:
            continue
        high_freq_id = int(high_freq_id)
        low_freq_id = int(low_freq_id)
        if freq_type == "high":
            remap[high_freq_id] = high_freq_id
            remap[low_freq_id] = high_freq_id
            reverse_map[high_freq_id] = [high_freq_id, low_freq_id]
            reverse_map[low_freq_id] = [high_freq_id, low_freq_id]
        else:
            remap[high_freq_id] = low_freq_id
            remap[low_freq_id] = low_freq_id
            reverse_map[high_freq_id] = [low_freq_id, high_freq_id]
            reverse_map[low_freq_id] = [low_freq_id, high_freq_id]

    return remap, reverse_map

# we assume that the vocab is id: 'word'
def remap_input_ids(input_ids, mapper):
    new_input_ids = input_ids
    cpy = input_ids
    for i, ids in enumerate(input_ids):
        # print(cpy[i])

        for j, token in enumerate(ids):
            if token == 0:
                continue
            #if token in utils.UNIQUE_TOKENS:
            #    continue

            new_input_ids[i][j] = mapper[int(token)]
        # print(new_input_ids[i])
        #exit(1)
    return new_input_ids

