
from collections import defaultdict
import string
import os
import regex as re
import random
import numpy as np
import utils
from numba import jit, prange
import logging as log
from tqdm import tqdm
#from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
import time
from transformers import AutoModel, AutoTokenizer
import pickle
import copy
import torch
import sklearn.metrics.pairwise as sklearn
from scipy.stats import gamma
import sentencepiece as spm
import lzma

"""
The logic of the mapper goes as follow:
I define a RemapBase class that defines the functionality of the remapper
it implements:
create_remap - that creates a dictionairy called remap and reverse remap
each key in this dictionairy is a token, and the value of the key is the new token we wil assign.
The reverse remap is just the opposite of the key ->value.
def remap_input_ids accepts token ids and the attention mask and returns new token ids that are remapped.

For testing/etc we have the print_noise_text.
Also, for the random and high/low frequency mappers we implement "remove forbidden tokens".
There are some tokens (chinese/japanese/korean) that were not mapped between roberta and the wiki training
so we just remove them and make sure we do not remap them (should have little effect)

"""
class RemapBase:
    def __init__(self):
        self.remap = {}
        self.modelname = ""
        self.reverse_map = {}
        self.fname = ""
        self.stop_words = {}
        self.special_tokens = []
        self.skip_stop_words = False
        pass

    def generate_word2token(self, text):
        # if roberta
        word2token = []
        if not self.skip_stop_words:
            return []

        if "roberta" in self.modelname:                
            pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
            for word in re.findall(pat, text):
                tokenized = self.tokenizer.encode(word, add_special_tokens=False)
                for t in tokenized:
                    word2token.append((word, t))

        else:
            pieces = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            words = []
            for p in pieces:
                words.append(p[0][1:])
            for i, word in enumerate(words):
                if word == "qnli" or (words[i-1] == "qnli" and word == "question:") or word == "sentence:":
                    continue                               
                else:
                    tokenized = self.tokenizer.encode(word, add_special_tokens=False)
                    for t in tokenized:
                        if t in self.tokenizer.all_special_ids:
                            continue
                        word2token.append((word, t))
        return word2token
    
    def get_noise_text(self, txt):
        tokenized_inputs = self.tokenizer.batch_encode_plus(
                [txt], pad_to_max_length=True, return_tensors="pt"
            )
        am = tokenized_inputs["attention_mask"][0]
        ids = tokenized_inputs["input_ids"][0]
        new_ids = self.apply_noise_to_input_ids(txt, ids, am)
        new_txt = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return new_txt
    
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

    def apply_noise_to_input_ids(self, text, ids, am):
        return ids
    
    def get_map(self):
        if self.remap == {}:
            self.create_remap()
        return self.remap

    def checkpoint(self, input_ids):
        with open(self.fname, "wb") as f:
            pickle.dump(input_ids, f)

    def restart(self):
        if os.path.exists(self.fname) and os.stat(self.fname).st_size > 1:
            # we already mapped this
            with open(self.fname, "rb") as f:
                print("Already remapped this:", self.fname)
                return pickle.load(f)
    
    def print_noise_text(self, tokenizer, text):
        pass
    
    def remap_sentence(self):
        pass

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
        return new_input_ids, attention_mask
    
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

    def find_stop_words(self, tokens):
        """
        We are going to skip these tokens :)
        """
        indices_to_skip = []
        for sw in self.stop_words:
            is_subset = all(item in list1 for item in subset_list)

            
        return indices_to_skip

class RemapNoise(RemapBase):
    def __init__(self, model, dataset, stop_words=False, eta=1):
        super().__init__()
        self.modelname = model
        self.skip_stop_words = stop_words
        model1 = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        model1.eval() # important!
        with torch.no_grad():
            if "t5" in model:
                self.word_embeddings = model1.encoder.embed_tokens.weight.numpy()
            else:
                self.word_embeddings = model1.embeddings.word_embeddings.weight.numpy() # matrix 50265x768, lookup table
            self.similarity_score = sklearn.cosine_similarity
        self.dataset = dataset
        self.special_tokens = self.tokenizer.all_special_ids
        self.eta = eta
        if "t5" in model:
            model = "t5"
        if stop_words:
            self.fname =  f"remap_noise_{model}_eta{self.eta}_{self.dataset}_skip"
        else:
            self.fname = f"remap_noise_{model}_eta{self.eta}_{self.dataset}"

    # def get_noise_text(self, tokenizer, text):
    #     tokens = utils.get_input_ids_from_text(tokenizer, text)
    #     new_tokens = copy.deepcopy(tokens)
    #     # print(text, new_tokens)
    #     for j in range(len(tokens)):
    #         if tokens[j] in self.special_tokens:
    #             continue
    #         new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
    #         new_embedding_vector = self.word_embeddings[tokens[j]] + self.get_noise()
    #         new_token = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=int(tokens[j]))
    #         new_tokens[j] = torch.tensor(new_token, dtype=torch.long)
    #     new_txt = utils.get_text_from_input_ids(tokenizer, new_tokens)
    #     return new_txt
        
    def print_noise_text(self, tokenizer, text):
        fname = f"print_remap_noise_{self.modelname}_eta{self.eta}_{self.dataset}_text.txt"  
        with open(fname, "w") as f:
            if isinstance(text, list):
                for i, txt in enumerate(text):
                    if i == 100:
                        exit(1)
                    new_txt = self.get_noise_text(tokenizer, txt)
                    print(txt)
                    print(new_txt)
                    print("-------------------------------------")
                    f.writelines("original text: " + txt + "\n")
                    f.writelines("new      text: " + new_txt +"\n") 
                    f.writelines("-----------------------------------------------------------------\n")
            else:
                new_txt = self.get_noise_text(tokenizer, text)    
                f.writelines("original text: " + text)
                f.writelines("new      text: " + new_txt)
                f.writelines("-----------------------------------------------------------------")

    def remap_input_ids(self, input_ids, attention_mask, name=""):
        print(self.fname)
        print("OUTDATED")
        exit(0)
        new_input_ids = self.restart()
        if new_input_ids == None:
            new_input_ids = copy.deepcopy(input_ids)
            print("Starting new pkl")
        with torch.no_grad():
            t1 = time.time()
        # cpy = input_ids
            for i, tokens in tqdm(enumerate(input_ids)):
                t2 = time.time()
                if all(new_input_ids[i].eq(input_ids[i])): # we haven't change it 
                    for j in range(len(tokens)):
                        if attention_mask[i][j] == 0 or tokens[j] in self.special_tokens:
                            continue
                        new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
                        new_embedding_vector = self.word_embeddings[tokens[j]] + self.get_noise()
                        new_token = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=int(tokens[j]))
                        new_input_ids[i][j] = torch.tensor(new_token, dtype=torch.long)
                    if i % 500 == 1:
                        self.checkpoint(new_input_ids)
                print('Time for sentence:', time.time() - t2, 'Time till now:', time.time() - t1)
        self.count_changed_tokens(input_ids, new_input_ids)

        with open(self.fname, "wb") as f:
            pickle.dump(new_input_ids, f)
        return new_input_ids, attention_mask
    
    def get_noise(self):
        r = gamma.rvs(self.word_embeddings[0].shape, scale=1/self.eta)
        p = utils.generate_points_within_hypersphere(num_points=1, num_dimensions=self.word_embeddings[0].shape, radius=1)
        rp = r * p
        return rp[0]
    
    def get_file_name(self):
        if self.fname == "":
            self.fname = f"remap_noise_{self.modename}_eta{self.eta}_{self.dataset}"
        return self.fname
    
    def apply_noise_to_input_ids(self, text, ids, am):
        new_ids = copy.deepcopy(ids)

        with torch.no_grad():
            word2token = self.generate_word2token(text)

            idx = 0
            for i, id in enumerate(ids):
                if am[i] == 0 or id in self.special_tokens:
                    continue
                # just making sure i'm not wrong and the tokenization process is the same
                if self.skip_stop_words:
                    original_word = word2token[idx][0]
                    if word2token[idx][1] != id:
                        print("Not same tokenization process", text)
                        print(ids)
                        print(word2token)
                        exit(1)
                idx = idx + 1
                if self.skip_stop_words and original_word.lstrip() in STOP_WORDS:
                    #self.stopwords_counter = self.stopwords_counter + 1 # maybe for later?
                    new_ids[i] = id
                    continue
                new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
                new_embedding_vector = self.word_embeddings[id] + self.get_noise()
                new_ids[i] = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=int(id))
        return new_ids
    
class RemapConv(RemapBase):
    def __init__(self, model, dataset, stencil_size=3, stride=1, sigma=1, tokenizer="", stopwords=False, punctuated=False):
        super().__init__()
        self.modelname = model
        model1 = AutoModel.from_pretrained(model)
        if tokenizer == "":
            self.tokenizer = AutoTokenizer.from_pretrained(model) # save toknizer, might use it later
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) # save toknizer, might use it later
        self.special_tokens = self.tokenizer.all_special_ids
        model1.eval() # important!
        # Get the word embedding table
        with torch.no_grad():
            if "t5" in model:
                self.word_embeddings = model1.encoder.embed_tokens.weight.numpy()
            else:
                self.word_embeddings = model1.embeddings.word_embeddings.weight.numpy() # matrix 50265x768, lookup table
            self.similarity_score = sklearn.cosine_similarity
        self.dataset = dataset # just for naming
        if stencil_size % 2 == 0:
            print("cannot stencil with even number")
            exit(1)
        self.skip_stop_words = stopwords
        self.sigma = sigma
        self.weights = utils.gaussian_weights(stencil_size, sigma=self.sigma)#0.6)
        
        self.indices = [0] * stencil_size
        start = int((stencil_size - 1) / 2)
        for i in range(stencil_size):
            self.indices[i] = i - start
        self.stride = stride # no use
        self.stencil_size = stencil_size
        self.is_punctuated = punctuated
        self.stopwords_counter = 0
        if "t5" in model:
            model = "t5"
            
        if self.is_punctuated:
            self.fname = f"remap_punctuatedconv_{model}_sigma{self.sigma}_{str(self.stencil_size)}_stride{str(self.stride)}_{self.dataset}"
        else:
            self.fname = f"remap_conv_{model}_sigma{self.sigma}_{str(self.stencil_size)}_stride{str(self.stride)}_{self.dataset}"
        if self.skip_stop_words:
            self.fname = self.fname + "_skipword"
        
        
    def apply_noise_to_input_ids(self, text, ids, am):
        new_ids = copy.deepcopy(ids)
        # if self.skip_stop_words:
        with torch.no_grad():

            word2token = self.generate_word2token(text)
            idx = 0
            for i, id in enumerate(ids):
                if am[i] == 0 or id in self.special_tokens:
                    continue
                # just making sure i'm not wrong
                if self.skip_stop_words:
                    original_word = word2token[idx][0]
                    if word2token[idx][1] != id:
                        print("Not same tokenization process", text)
                        print(word2token)
                        print(ids)
                        print(ids[i])
                        print(word2token[idx][0], word2token[idx][1])
                        print(am[i])
                        print(idx, i)
                        exit(1)
                idx = idx + 1
                if self.skip_stop_words and original_word.lstrip() in STOP_WORDS:
                    self.stopwords_counter = self.stopwords_counter + 1 # maybe for later?
                    new_ids[i] = id
                    continue
                tokens_to_fuse, weights, _ = self.get_tokens_and_weights(ids, i)
                new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
                for token, weight in zip(tokens_to_fuse, weights):
                    new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight
                if self.is_punctuated:
                    skipper = -1
                else:
                    skipper = int(id)
                new_ids[i] = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=skipper)
            return new_ids

    # Here we report as an np array the tokenis that we are going to fuse along with their weights
    def get_tokens_and_weights(self, tokens, j):
        tokens_to_fuse = []
        weights = []
        weight_skipped = []
        indices = []
        for i, ind in enumerate(self.indices):
            # Dealing with first or last tokens...... or tokens that are unwanted such as 1,2
            if j + ind >= len(tokens) or j + ind < 0 or tokens[j+ind] in self.special_tokens or (ind == 0 and self.is_punctuated):
                weight_skipped.append(self.weights[i])
            else:
                tokens_to_fuse.append(tokens[j + ind])
                weights.append(self.weights[i])
                indices.append(j + ind)
        additional_weight = np.sum(weight_skipped) / (self.stencil_size - len(weight_skipped))
        weights = weights + additional_weight
        if np.sum(weights) < 0.999999 or np.sum(weights) > 1.000001:
            print("Error with weights")
            exit(1)
        return np.array(tokens_to_fuse), np.array(weights), np.array(indices)
    
    def get1_noise_text(self, txt):
        tokenized_inputs = self.tokenizer.batch_encode_plus(
                [txt], pad_to_max_length=True, return_tensors="pt"
            )
        am = tokenized_inputs["attention_mask"][0]
        ids = tokenized_inputs["input_ids"][0]
        new_ids = self.apply_noise_to_input_ids(txt, ids, am, True)
        new_txt = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        #print(new_txt)
        return new_txt
        # tokens = utils.get_input_ids_from_text(self.tokenizer, txt)
        if not isinstance(txt, str):
            tokens = txt
        else:
            tokens = self.tokenizer.tokenize(txt)
        new_tokens = copy.deepcopy(tokens)
        for j in range(len(tokens)):
            if tokens[j] in self.special_tokens: # tokens that we do not want to convert
                continue
            tokens_to_fuse, weights, _ = self.get_tokens_and_weights(tokens, j) # if we want a different kind of weights
            new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
            for token, weight in zip(tokens_to_fuse, weights):
                new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight
            # skip tells us to always return a different token
            if self.is_punctuated:
                skipper = -1
            else:
                skipper = int(tokens[j])
            new_token = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=skipper)
            new_tokens[j] = torch.tensor(new_token, dtype=torch.long)
        new_txt = self.tokenizer.decode(new_tokens, skip_special_tokens=True)#utils.get_text_from_input_ids(self.tokenizer, new_tokens)
        return new_txt, tokens, new_tokens

    def print_noise_text(self, tokenizer, text):
        fname = f"print_remap_conv_{self.modelname}_sigma{self.sigma}_{str(self.stencil_size)}_stride{str(self.stride)}_{self.dataset}_text.txt" #f"remap_noise_eta{self.eta}_{self.dataset}_text.txt"
        with open(fname, "w") as f:
            if isinstance(text, list):
                for i, txt in enumerate(text):
                    if i == 100:
                        exit(1)
                    new_txt, t1, t2 = self.get_noise_text(tokenizer, txt)
                    f.writelines("original text: " + txt + "\n")
                    f.writelines("new      text: " + new_txt +"\n")
                    f.writelines("original token: " + str(t1) + "\n")
                    f.writelines("new      token: " + str(t2) + "\n")
                    f.writelines("-----------------------------------------------------------------\n")
                    print(txt, new_txt)
            else:
                new_txt = self.get_noise_text(tokenizer, text)
                f.writelines("original text: " + text)
                f.writelines("new      text: " + new_txt)
                f.writelines("-----------------------------------------------------------------")

    def get_new_tokens(self, tokens, am):
        new_tokens = copy.deepcopy(tokens)
        for i, token in enumerate(tokens):
            if am[i] == 0 or tokens[i] in self.special_tokens:
                continue
            tokens_to_fuse, weights, _ = self.get_tokens_and_weights(tokens, i) # if we want a different kind of weights
            new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
            for token, weight in zip(tokens_to_fuse, weights):
                new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight
            if self.is_punctuated:
                skipper = -1
            else:
                skipper = int(tokens[i])
            new_tokens[i] = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=skipper)
        return new_tokens

    def remap_by_text(self, texts, name):
        #self.fname = f"remap_conv_{self.modelname}_sigma{self.sigma}_{str(self.stencil_size)}_{self.dataset}_{name}.pkl"
        fname2 = f"remap_conv_{self.modelname}_sigma{self.sigma}_{str(self.stencil_size)}_{self.dataset}_{name}.txt"
        if os.path.exists(self.fname) and os.stat(self.fname).st_size > 1:
            # we already mapped this
            with open(self.fname, "rb") as f:
                print("Already remapped this:", self.fname)
                new_text = pickle.load(f)
        
        new_text = []
        for i, sentence in enumerate(texts):
            words = sentence.split()
            tokens = self.tokenizer.tokenize(words, add_special_tokens=False)
            new_tokens = []
            for j, token in enumerate(tokens):
                tokens_to_fuse, weights, _ = self.get_tokens_and_weights(tokens, j)
                new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
                for token, weight in zip(tokens_to_fuse, weights):
                    new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight
                new_token = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=int(tokens[j]))
                new_tokens.append(new_token)
            new_text.append(self.tokenizer.decode(new_tokens))
            if i % 500 == 1:
                self.checkpoint(new_text)
                with open(fname2, "w") as f:
                    f.writelines(new_text)
    
        
    def remap_input_ids(self, input_ids, attention_mask, name=""):
        # if self.stride != 1:
        #     self.fname = f"remap_conv_{self.modelname}_sigma{self.sigma}_{str(self.stencil_size)}_stride{str(self.stride)}_{self.dataset}_{name}.pkl"
        # else:
        #     self.fname = f"remap_conv_{self.modelname}_sigma{self.sigma}_{str(self.stencil_size)}_{self.dataset}_{name}.pkl"
        # print(self.fname)
        tokenized_inputs["input_ids"][0] = self.remapper.apply_noise_to_input_ids(line, tokenized_inputs["input_ids"][0], attention_mask)
        new_input_ids = self.restart()
        if new_input_ids == None:
            new_input_ids = copy.deepcopy(input_ids)
            print("Starting new pkl")
        with torch.no_grad():
            t1 = time.time()
        # cpy = input_ids
            for i, tokens in tqdm(enumerate(input_ids)):
                changed = -1
                t2 = time.time()
                cnt = 0
                if all(new_input_ids[i].eq(input_ids[i])): # we haven't change it 
                    for j in range(len(tokens)):
                        #cnt = cnt + 1
                        if attention_mask[i][j] == 0 or tokens[j] in self.special_tokens:
                            continue

                        tokens_to_fuse, weights, _ = self.get_tokens_and_weights(tokens, j) # if we want a different kind of weights
                        new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
                        for token, weight in zip(tokens_to_fuse, weights):
                            new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight
                        new_token = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=int(tokens[j]))
                        new_input_ids[i][j] = torch.tensor(new_token, dtype=torch.long)
                        cnt = cnt + 1
                    if i % 500 == 1:
                        self.checkpoint(new_input_ids)
                #print("Time for sentence:", -t2 + time.time(), 'Avg time per token:', (-t2 +time.time() )/cnt, "Time till now:", -t1 + time.time())
        
        self.count_changed_tokens(input_ids, new_input_ids)

        with open(self.fname, "wb") as f:
            pickle.dump(new_input_ids, f)
        return new_input_ids, attention_mask
    
    def get_file_name(self):
        return self.fname
    

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

    def print_noise_text(self, tokenizer, text):
        fname = "print_remap_random_sst2.txt"
        self.create_remap()
        #tokens = utils.get_input_ids_from_text(tokenizer, txt)
        #new_tokens = copy.deepcopy(tokens)
        with open(fname, 'w') as f:
            for i, txt in enumerate(text):
                if i == 100:
                    exit(1)
                new_txt =  new_txt = self.get_noise_text(tokenizer, txt)
                print(txt)
                print(new_txt)
                print("-------------------------------------")
                f.writelines("original text: " + txt + "\n")
                f.writelines("new      text: " + new_txt + "\n")
                f.writelines("-----------------------------------------------------------------\n")
#self.get_noise_text(tokenizer, txt)

    def get_noise_text(self,tokenizer, txt):
        tokens = utils.get_input_ids_from_text(tokenizer, txt)
        new_tokens = copy.deepcopy(tokens)
        for j in range(len(tokens)):
            if tokens[j] == 0 or tokens[j] == 2:
                continue
            new_tokens[j] = self.remap[int(tokens[j])]
            # tokens_to_fuse, weights = self.get_tokens_and_weights(tokens, j) # if we want a different kind of weights
            # new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
            #for token, weight in zip(tokens_to_fuse, weights):
            #    new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight
            #new_token = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=int(tokens[j]))
            #new_tokens[j] = torch.tensor(new_token, dtype=torch.long)
        new_txt = utils.get_text_from_input_ids(tokenizer, new_tokens)
        return new_txt

                
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
    

    
    def print_noise_text(self, tokenizer, text):
        fname = "print_remap_freq_sst2.txt"
        self.create_remap()
        #tokens = utils.get_input_ids_from_text(tokenizer, txt)
        #new_tokens = copy.deepcopy(tokens)
        with open(fname, 'w') as f:
            for i, txt in enumerate(text):
                if i == 100:
                    exit(1)
                new_txt =  new_txt = self.get_noise_text(tokenizer, txt)
                print(txt)
                print(new_txt)
                print("-------------------------------------")
                f.writelines("original text: " + txt + "\n")
                f.writelines("new      text: " + new_txt + "\n")
                f.writelines("-----------------------------------------------------------------\n")
#self.get_noise_text(tokenizer, txt)

    def get_noise_text(self,tokenizer, txt):
        tokens = utils.get_input_ids_from_text(tokenizer, txt)
        new_tokens = copy.deepcopy(tokens)
        for j in range(len(tokens)):
            if tokens[j] == 0 or tokens[j] == 2:
                continue
            new_tokens[j] = self.remap[int(tokens[j])]
            # tokens_to_fuse, weights = self.get_tokens_and_weights(tokens, j) # if we want a different kind of weights
            # new_embedding_vector = np.full(self.word_embeddings[0].shape, 0.0)
            #for token, weight in zip(tokens_to_fuse, weights):
            #    new_embedding_vector = new_embedding_vector + self.word_embeddings[token] * weight
            #new_token = utils.get_most_similar_token(self.word_embeddings, new_embedding_vector, skip=int(tokens[j]))
            #new_tokens[j] = torch.tensor(new_token, dtype=torch.long)
        new_txt = utils.get_text_from_input_ids(tokenizer, new_tokens)
        return new_txt

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

