from collections import defaultdict
import string
import os
import re
import random
import numpy as np
import utils
from numba import jit, prange
import logging as log
from tqdm import tqdm
import time
from transformers import AutoModel
import pickle
import copy
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
    def get_file_name(self):
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
    def remap_input_ids(self, input_ids, attention_mask, name=""):
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
    
    
    def count_changed_tokens(self, old_input_ids, new_input_ids):
        survived_tokens = 0
        total_tokens = 0 
        for i, ids in enumerate(old_input_ids):
            for j, token in enumerate(ids):
                if int(token) == 0 or token == 1 or token == 2:
                    continue
                if int(old_input_ids[i][j]) == int(new_input_ids[i][j]):
                    survived_tokens = survived_tokens + 1
                total_tokens = total_tokens + 1
        print("survived tokens:", survived_tokens, survived_tokens/ total_tokens)

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
    def __init__(self, model, dataset, stencil_size=3, stride=1):
        super().__init__()
        model1 = AutoModel.from_pretrained(model)
        model1.eval() # important!
        with torch.no_grad():
            self.word_embeddings = model1.embeddings.word_embeddings.weight.numpy() # matrix 50265x768, lookup table
            self.similarity_score = sklearn.cosine_similarity
        self.dataset = dataset
        if stencil_size % 2 == 0:
            print("cannot stencil with even number")
            exit(1)
        self.weights = utils.gaussian_weights(stencil_size, sigma=0.6)
        self.indices = [0] * stencil_size
        start = int((stencil_size - 1) / 2)
        for i in range(stencil_size):
            self.indices[i] = i - start
        self.fname = ""
        self.stride = stride
        self.stencil_size = stencil_size
    
    def get_tokens_and_weights(self, tokens, j):
        tokens_to_fuse = []
        weights = []
        weight_skipped = []
        for i, ind in enumerate(self.indices):
            if j + ind >= len(tokens) or j + ind < 0 or tokens[j+ind] == 0 or tokens[j+ind] == 1 or tokens[j+ind] == 2:
                weight_skipped.append(self.weights[i])
            else:
                tokens_to_fuse.append(tokens[j + ind])
                weights.append(self.weights[i])
        additional_weight = np.sum(weight_skipped) / (self.stencil_size - len(weight_skipped))
        weights = weights + additional_weight
        if np.sum(weights) < 0.999999 or np.sum(weights) > 1.000001:
            print("Error with weights")
            exit(1)
        return np.array(tokens_to_fuse), np.array(weights)
    
    def remap_input_ids(self, input_ids, attention_mask, name=""):
        if self.stride != 1:
            self.fname = f"remap_conv_{str(self.stencil_size)}_stride{str(self.stride)}_{self.dataset}_{name}.pkl"
        else:
            self.fname = f"remap_conv_{str(self.stencil_size)}_{self.dataset}_{name}.pkl"
        print(self.fname)
        
        new_input_ids = self.restart()
        if new_input_ids == None:
            new_input_ids = copy.deepcopy(input_ids)
            print("Starting new pkl")
        with torch.no_grad():
        # cpy = input_ids
            for i, tokens in tqdm(enumerate(input_ids)):
                changed = -1
                if all(new_input_ids[i].eq(input_ids[i])): # we haven't change it 
                    print("Restarting in:", i , "out of:", len(input_ids))
                    for j in range(len(tokens)):
                        if attention_mask[i][j] == 0 or tokens[j] == 0 or tokens[j] == 1 or tokens[j] == 2:
                            continue
                        changed = changed + 1
                        if changed % self.stride != 0:
                            continue
                        tokens_to_fuse, weights = self.get_tokens_and_weights(tokens, j) # if we want a different kind of weights
                        new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
                        for token, weight in zip(tokens_to_fuse, weights):
                            new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight
                        new_token = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=int(tokens[j]))
                        #print("Step:", i, "New token:", new_token, "Old token:", tokens[j])
                        new_input_ids[i][j] = torch.tensor(new_token, dtype=torch.long)
                    if i % 500 == 1:
                        self.checkpoint(new_input_ids)
        
        self.count_changed_tokens(input_ids, new_input_ids)

        with open(self.fname, "wb") as f:
            pickle.dump(new_input_ids, f)
        return new_input_ids
    
    def get_file_name(self):
        return self.fname
    
    def checkpoint(self, input_ids):
        with open(self.fname, "wb") as f:
            pickle.dump(input_ids, f)

    def restart(self):
        if os.path.exists(self.fname) and os.stat(self.fname).st_size > 1:
            # we already mapped this
            with open(self.fname, "rb") as f:
                print("Already remapped this:", self.fname)
                return pickle.load(f)


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

