from collections import defaultdict
import string
import os
import re
import random
import numpy as np
import utils
import logging as log
from tqdm import tqdm
from transformers import AutoModel
import pickle
import torch
import sklearn.metrics.pairwise as sklearn

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
    def remap_input_ids(self, input_ids, attention_mask):
        survived_tokens = 0
        total_tokens = 0 
        if self.remap == {}:
            self.create_remap()
        new_input_ids = input_ids
        # cpy = input_ids
        for i, ids in enumerate(input_ids):
            for j, token in enumerate(ids):
                if token == 0:
                    continue
                if int(token) == self.remap[int(token)]:
                    survived_tokens = survived_tokens + 1
                new_input_ids[i][j] = self.remap[int(token)]
                total_tokens = total_tokens + 1
        print("survived tokens:", survived_tokens, survived_tokens/ total_tokens)
        return new_input_ids
    
    def remove_forbidden_tokens(self, indices_to_shuffle, forbid):
        if forbid:
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
            indices_to_shuffle.pop(50264)
            indices_to_shuffle.pop(2)
            indices_to_shuffle.pop(1)
            indices_to_shuffle.pop(0)
            self.remap[2] = 2
            self.remap[50264] = 50264
            self.remap[1] = 1
            self.remap[0] = 0

class RemapConv(RemapBase):
    def __init__(self, model):
        super().__init__()
        model1 = AutoModel.from_pretrained(model)
        self.word_embeddings = model1.embeddings.word_embeddings.weight # matrix 50265x768, lookup table
        self.similarity_score = sklearn.cosine_similarity

    def get_unique(self):
        return "conv"
    
    def remap_input_ids(self, input_ids, attention_mask):
        survived_tokens = 0
        total_tokens = 0 
        new_input_ids = input_ids
        # cpy = input_ids
        for i, tokens in enumerate(input_ids):
            for j in range(len(tokens)):
                tokens_to_fuse, weights = self.get_tokens_and_weights(tokens, j)
                new_embedding_vector = []
                for token, weight in zip(tokens_to_fuse, weights):
                    new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight


        return new_input_ids
    
    def get_tokens_and_weights(self, tokens, j):
        tokens_to_fuse = []
        weights = []
        if j != 0:
            tokens_to_fuse.append[tokens[j-1]]
            weights.append(0.25)
        tokens_to_fuse.append(tokens[j])
        weights.append(0.5)
        if tokens[j+1] != 0:
            tokens_to_fuse.append[tokens[j+1]]
            weights.append(0.25)
        return tokens_to_fuse, weights

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

        self.remove_forbidden_tokens(indices_to_shuffle, self.forbid)
        
        if self.shuffle:
            np.random.seed(0)
            random.shuffle(indices_to_shuffle)
        while indices_to_shuffle != []:
            tokens = []
            token1 = indices_to_shuffle.pop()
            self.remap[token1] = token1
            tokens.append(token1)
            for i in range(self.remap_count - 1):
                if indices_to_shuffle == []: # odd vocab size
                    break
                token = indices_to_shuffle.pop()
                self.remap[token] = token1
                tokens.append(token)
            for t in tokens:
                self.reverse_map[t] = tokens
                
        #log.info(self.reverse_map)
        #log.info(self.remap)
        #exit(1)

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
        self.remove_forbidden_tokens(self.freq_ids, self.forbid)
        
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

