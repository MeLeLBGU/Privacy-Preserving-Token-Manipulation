from collections import defaultdict
import string
import os
import re
import random
import numpy as np
import utils
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
    
    def get_unique(self):
        return ""
    
    def get_reversed_map(self):
        if self.reverse_map == {}:
            self.create_remap()
            log.error("Bad use of get reversed map")
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
    def __init__(self, vocab, shuffle=True, forbidden_tokens=False, remap_count=2):
        super().__init__()
        log.info("You chose to remap the inputs ids randomly")
        self.vocab = vocab
        self.remap_count = remap_count
        self.shuffle = shuffle
        self.forbid = forbidden_tokens
        self.forbidden_tokens = []
    
    def get_unique(self):
        return "random"
    
    def create_remap(self):
        np.random.seed(0)

        # indices_to_shuffle = [i for i in range(len(self.vocab)) if (i % self.remap_count != 0)]
        indices_to_shuffle = [i for i in range(len(self.vocab))]
        self.remap = {int(key): 0 for key in self.vocab}
        self.reverse_map = {int(key): None for key in self.vocab}

        if self.forbid:
            # there are some tokens that we do not want to map because they will cause incosisntency
            a_file = open("roberta_gpt_mapper.pkl", "rb")
            roberta_gpt_mapper = pickle.load(a_file)
            forbidden = []
            for i in range(len(roberta_gpt_mapper)):
                if roberta_gpt_mapper[i] == -1:
                    forbidden.append(i)
                    self.remap[i] = i
            forbidden.sort(reverse=True)
            for rm in forbidden:
                indices_to_shuffle.pop(rm)
            a_file.close()
        else:
            # we dont want to remap the special characters
            indices_to_shuffle.pop(50264)
            indices_to_shuffle.pop(2)
            indices_to_shuffle.pop(1)
            indices_to_shuffle.pop(0)
            self.remap[2] = 2
            self.remap[50264] = 50264
            self.remap[1] = 1
            self.remap[0] = 0
        
        if self.shuffle:
            np.random.seed(0)
            random.shuffle(indices_to_shuffle)
        
        while indices_to_shuffle != []:
            token1 = indices_to_shuffle.pop()
            if indices_to_shuffle == []: # odd vocab size
                self.remap[token1] = token1
                self.reverse_map[token1] = [token1, token1]
                break
            token2 = indices_to_shuffle.pop()
            self.remap[token1] = token1
            self.remap[token2] = token1
            self.reverse_map[token1] = [token1, token2]
            self.reverse_map[token2] = [token1, token2]
        # for test
        log.info(self.reverse_map)
        # log.info("Remap[1000 ] =" + str(self.remap[1000]))
        # log.info("ReverseRemap[1000] =" + str(self.reverse_map[1000]))


class RemapFrequency(RemapBase):
    def __init__(self, vocab, freq_path, freq_type, forbid=False, window="all"):
        super().__init__()
        if "high" in freq_type:
            self.freq_type = "high"
        elif "low" in freq_type:
            self.freq_type = "low"
        self.window = window

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
        self.forbid = forbid
        self.forbidden_tokens = []
    
    def get_unique(self):
        return "freq" + self.freq_type + self.window
    
    def create_remap(self):
        self.freq_ids = sorted(self.freq_ids.items(), key=lambda x: x[0], reverse=False)

        self.remap = {int(key): 0 for key in self.vocab}
        self.reverse_map = {int(key): None for key in self.vocab}
        if self.forbid:
            # there are some tokens that we do not want to map because they will cause incosisntency
            a_file = open("roberta_gpt_mapper.pkl", "rb")
            roberta_gpt_mapper = pickle.load(a_file)
            for i in range(len(roberta_gpt_mapper)):
                if roberta_gpt_mapper[i] == -1:
                    self.forbidden_tokens.append(i)
                    self.remap[i] = i
            self.forbidden_tokens.sort(reverse=True)
            for rm in self.forbidden_tokens:
                self.freq_ids.pop(rm)
            a_file.close()
        else:
            # we dont want to remap the special characters
            self.freq_ids.pop(50264)
            self.freq_ids.pop(2)
            self.freq_ids.pop(1)
            self.freq_ids.pop(0)
            self.remap[2] = 2
            self.remap[50264] = 50264
            self.remap[1] = 1
            self.remap[0] = 0
        
        sorted_freq_ids = sorted(self.freq_ids, key=lambda x: x[1], reverse=True)

        #for i in tqdm(range(int(len(self.vocab)/2))):
        while sorted_freq_ids != []:

            high_freq_id, _ = sorted_freq_ids.pop(0)
            low_id = self.get_low_freq_index(high_freq_id)
            while low_id >= len(sorted_freq_ids):
                low_id = low_id - 1
            if sorted_freq_ids == []: # odd vocab size
                self.remap[high_freq_id] = high_freq_id
                self.reverse_map[high_freq_id] = [high_freq_id, high_freq_id]
                break
            low_freq_id,_ = sorted_freq_ids.pop(low_id)

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
        log.info(self.remap)
        log.info("Remap[10] =" + str(self.remap[10]))
        log.info("ReverseRemap[10] =" + str(self.reverse_map[10]))
        

    def get_low_freq_index(self, i):
        if self.window == "all":
            return -1
        elif self.window == "half":
            return int(i + (len(self.freq_ids)/2))
        else:
            return i + int(self.window) 

