from collections import defaultdict
import string
import os
import tokenizers
import re
from transformers import BertTokenizer
import random
import utils
import collections
import logging as log
from tqdm import tqdm
import pickle


class RemapBase:
    def __init__(self):
        self.remap = {}
        self.reverse_map = {}
        pass

    def create_remap(self):
        # implement the remap
        pass

    def get_reversed_map(self):
        if self.reverse_map == {}:
            self.create_remap()
        return self.reverse_map

    def get_map(self):
        if self.remap == {}:
            self.create_remap()
        return self.remap

    # we assume that the vocab is id: 'word'
    def remap_input_ids(self, input_ids):
        if self.remap == {}:
            self.create_remap()
        new_input_ids = input_ids
        cpy = input_ids
        for i, ids in enumerate(input_ids):
            for j, token in enumerate(ids):
                if token == 0:
                    continue

                new_input_ids[i][j] = self.remap[int(token)]
            # print(new_input_ids[i])
            # exit(1)
        return new_input_ids


class RemapRandom(RemapBase):
    def __init__(self, vocab, shuffle=True, remap_count=2):
        super().__init__()
        log.info("You chose to remap the inputs ids randomly")

        self.remap_count = remap_count
        self.shuffle = shuffle

    def create_remap(self):
        save_vocab = self.vocab.copy()
        new_vocab = self.vocab
        indices_to_shuffle = [i for i in range(len(self.vocab)) if (i % remap_count != 0)]

        if self.shuffle:
            random.shuffle(indices_to_shuffle)
        k = 0
        for i, token in enumerate(tqdm(self.vocab)):
            if i % self.remap_count == 0:
                lst = []
                new_token = token
                lst.append(i)  # for the reverse vocabulary
            else:
                self.remap[indices_to_shuffle[k]] = new_token
                lst.append(indices_to_shuffle[k])
                k = k + 1
            if (i + 1) % self.remap_count == 0:
                self.reverse_map[int(new_token)] = lst

class RemapFrequency(RemapBase):
    def __init__(self, vocab, freq_path, freq_type):
        super().__init__()
        if "high" in freq_type:
            self.freq_type = "high"
        elif "low" in freq_type:
            self.freq_type = "low"

        self.vocab = vocab
        if not os.path.exists(freq_path):
            log.error("No frequency dictionary path")
            exit(1)

        a_file = open(freq_path, "rb")
        self.freq_ids = pickle.load(a_file)
        if self.freq_ids == {}:
            log.error("No frequency dictionary path")
            exit(1)
        a_file.close()

    def create_remap(self):
        sorted_freq_ids = sorted(self.freq_ids.items(), key=lambda x: x[1])
        self.remap = {int(key): 0 for key in vocab}
        self.reverse_map = {int(key): None for key in vocab}

        for i, tpl in enumerate(tqdm(sorted_freq_ids)):
            high_freq_id, _ = sorted_freq_ids[-(i + 1)]
            low_freq_id, _ = sorted_freq_ids[i]
            
            
            high_freq_id = int(high_freq_id)
            low_freq_id = int(low_freq_id)
            if self.freq_type == "high":
                self.remap[high_freq_id] = high_freq_id
                self.remap[low_freq_id] = high_freq_id
                self.reverse_map[high_freq_id] = [high_freq_id, low_freq_id]
                self.reverse_map[low_freq_id] = [high_freq_id, low_freq_id]
            else:
                self.remap[high_freq_id] = low_freq_id
                self.remap[low_freq_id] = low_freq_id
                self.reverse_map[high_freq_id] = [low_freq_id, high_freq_id]
                self.reverse_map[low_freq_id] = [low_freq_id, high_freq_id]



