from collections import defaultdict
import string
import os
import re
import random
import utils
from scipy.spatial.distance import cdist
import scipy
import numpy as np
import utils
from numba import jit, prange
import logging as log
from tqdm import tqdm
import time
from transformers import AutoModel
import sklearn.metrics.pairwise as sklearn
import pickle
import copy
import torch
import sklearn.metrics.pairwise as sklearn
import tqdm
import argparse
import os
import sys
import copy
import json

def checkpoint(attacker_file, i, dict1):
    if i % 10 == 0:
        with open(attacker_file, "r+") as outfile:
            json.dump(dict1, outfile, indent=4)

def restart(attacker_file):
    last_index = -1
    dict1 = {}
    if os.path.exists(attacker_file) and os.stat(attacker_file).st_size > 5:
        with open(attacker_file, "r") as f:
            dict1 = json.load(f)
            last_index = int(list(dict1.keys())[-1])
    else:
        f = open(attacker_file, "w")
        f.close()
    return dict1, last_index

# def knn_attack(victim_model, tokenizer, eval_dataloader, label_list, attack_layer=3, emb=None, is_token=False, topk=1):
#     device = victim_model.device
#     special_tokens = [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]
#     metric_list = ['token_hit', 'rouge']
#     if is_token:
#         metric_list.append('ent_hit')
#     metric_counter = MetricCounter(metric_list)
#     if emb == None:
#         emb = copy.deepcopy(victim_model.bert.embeddings.word_embeddings.weight)
    
#     for batch in tqdm.tqdm(eval_dataloader):
#         batch = {key:value.to(device) for key,value in batch.items()}
#         labels = batch.pop('labels')
#         batch['output_hidden_states'] = True
#         with torch.no_grad():
#             outputs = victim_model(**batch)
#         masks = batch['attention_mask'].tolist()
#         batch_size = batch['input_ids'].shape[0]

#         for i in range(batch_size):
#             seq_len = len(masks[i]) if 0 not in masks[i] else masks[i].index(0)
#             hidden_states = outputs.hidden_states[attack_layer][i]

#             ed = torch.cdist(hidden_states, emb, p=2.0)
#             candidate_token_ids_topk = torch.topk(ed,topk,largest=False)[1]

#             temp_hit, temp_total = rouge(batch['input_ids'][i].unsqueeze(0), candidate_token_ids_topk[:,0].unsqueeze(0), tokenizer, special_tokens)
#             metric_counter.update('rouge', temp_hit, temp_total)
#             temp_hit, temp_total = token_hit(batch['input_ids'][i].unsqueeze(0), candidate_token_ids_topk[:,0].unsqueeze(0), tokenizer, special_tokens)
#             metric_counter.update('token_hit', temp_hit, temp_total)
#             if is_token:
#                 temp_hit, temp_total = ent_hit(batch['input_ids'][i].unsqueeze(0), candidate_token_ids_topk[:,0].unsqueeze(0), tokenizer, special_tokens, label_ids=labels[i].unsqueeze(0), id2label={idx:item for idx, item in enumerate(label_list) })
#                 metric_counter.update('ent_hit', temp_hit, temp_total)
#     attack_results = {metric: metric_counter(metric) for metric in metric_list}
#     return attack_results
def remove_non_zero_values(input_list):
    return [value for value in input_list if value != 0]

def stencil_attacker(real_tokens, tokens, am, emb_table, remapper):
    
    word_embeddings = remapper.word_embeddings
    # embeddings = np.zeros(len(tokens))
    # for i, token in enumerate(tokens):
    #     embeddings[i] = word_embeddings[token]
    # ones = (am == 1).sum()
    tokens1 = tokens * am
    tokens1 = remove_non_zero_values(tokens1)
    ones = len(tokens1) - 2
    #if ones > 15:

    #    print(ones)
    #    return [1], [1], [1]
    stencil_size = remapper.stencil_size
    A = np.zeros([ones, ones])
    b = np.zeros([ones, emb_table.shape[1]]) #? maybe..?        
    #print(tokens1)
    for i in range(ones):
        b[i] = emb_table[tokens1[i]]
        tokens_to_fuse, weights, indices = remapper.get_tokens_and_weights(tokens1, i)
        for j, ind in enumerate(indices): 
                A[i][ind] = weights[j]
    #print(A)
    x = np.linalg.solve(A, b)
    #print(x.shape)
    is_top5 = []
    is_top1 = []
    top5_tokens = []
    top1_tokens = []
    # top5_unorder = [False]*len(is_top1)
    j = 0
    for i, token in enumerate(tokens):
        if i + 1 == 512:
           continue
        if am[i] == 0 or i == 0 or am[i+1] == 0:
            continue
        similarity = get_similarity2(x[j], emb_table) # eucl space
        j = j + 1
        top5 = sorted(range(len(similarity)), key=lambda sub: similarity[sub])[1:6]
        top5_tokens.append(top5)
        is_top1.append(int(real_tokens[i]) == int(top5[0]))
        is_top5.append(real_tokens[i] in top5)
    top5_unorder = [False] * len(is_top1)
    for i, token in enumerate(real_tokens):
        if i + 1 == 512:
            continue
        if am[i] == 0 or i == 0 or am[i+1] == 0:
            continue
        for j, top5 in enumerate(top5_tokens):
            if token in top5:
                top5_unorder[j] = True

    return is_top1, is_top5, top5_unorder    

def knn_attack(real_tokens, tokens, am, emb_table):
    # todo
    is_top5 = []
    is_top1 = []
    top5_tokens = []
    top1_tokens = []
    # top5_unorder = [False]*len(is_top1)
    for i, token in enumerate(tokens):
        if i + 1 == 512:
           continue
        if am[i] == 0 or i == 0 or am[i+1] == 0:
            continue
        similarity = get_similarity(token, emb_table) # eucl space
        top5 = sorted(range(len(similarity)), key=lambda sub: similarity[sub])[1:6]
        top5_tokens.append(top5)
        is_top1.append(int(real_tokens[i]) == int(top5[0]))
        is_top5.append(real_tokens[i] in top5)
    top5_unorder = [False] * len(is_top1)
    for i, token in enumerate(real_tokens):
        if i + 1 == 512:
            continue
        if am[i] == 0 or i == 0 or am[i+1] == 0:
            continue
        for j, top5 in enumerate(top5_tokens):
            if token in top5:
                top5_unorder[j] = True

    return is_top1, is_top5, top5_unorder    
        

def get_similarity(token, emb_table):
    emb_token = np.zeros((1, emb_table.shape[1]))
    emb_token[0,:] = emb_table[token]
    #print(emb_token)
    a = utils.eucl_naive(emb_token, emb_table)[0]
    #a = utils.fast_cosine_matrix(emb_token[0], emb_table)#[0]
    #print(a,b)
    return a #utils.eucl_naive(emb_token, emb_table)[0]  

def get_similarity2(emb_token, emb_table):
    #print(emb_token)
    #a = utils.eucl_naive(emb_token, emb_table)[0]
    a = utils.fast_cosine_matrix(emb_token, emb_table)#[0]
    #print(a,b)
    return a #utils.eucl_naive(emb_token, emb_table)[0]  


def create_embedding_similarity_table(emb_table, cosine=True):
    similarity_table = np.zeros([len(emb_table), len(emb_table)])
    tokens = [i for i in range(len(emb_table))]
    if not cosine:
        for i,token in tqdm.tqdm(enumerate(tokens)):
            emb_token = np.zeros((1, emb_table.shape[1]))
            emb_token[0,:] = emb_table[token]
            similarity_table[i] = utils.eucl_naive(emb_token, emb_table)[0]  
    else:
        for i, token in enumerate(tokens):
            similarity_table[i] = sklearn.cosine_similarity([emb_table[token]], emb_table, "euclidean")[0]  
    return similarity_table
    

def main(model, tokenizer, am, mapped_train_input_ids, original_train_input_ids, attacker_file, remapper):
    dict1, last_index = restart(attacker_file)
    model1 = AutoModel.from_pretrained(model)
    model1.eval() # important!
    print("nn attacker file:", attacker_file)
    
    with torch.no_grad():
        if "t5" in model:
            emb_table = model1.encoder.embed_tokens.weight.numpy()
        else:
            emb_table = model1.embeddings.word_embeddings.weight.numpy() # matrix 50265x768, lookup table
        # similarity_table = create_embedding_similarity_table(emb_table)
    
    for i, tokens in enumerate(mapped_train_input_ids):
        if i > 10000:
            exit(1)
        if i <= last_index:
            continue
        original_ids = original_train_input_ids[i].tolist()
        checkpoint(attacker_file, i, dict1)
        t1 = time.time()
        is_top1, is_top5, top5_unorder = knn_attack(original_ids, tokens, am[i], emb_table)
        #is_top1, is_top5, top5_unorder = stencil_attacker(original_ids, tokens, am[i], emb_table, remapper)
        original_text = utils.get_text_from_input_ids(tokenizer, original_ids)
        # print("Top1 Reconstructed:", all(is_top1))
        # print("Top5 Reconstructed:", all(is_top5))
        # print("How many tokens in top1:", np.sum(is_top1), np.sum(is_top1) / len(is_top1))
        # print("How many tokens in top5:", np.sum(is_top5), np.sum(is_top5) / len(is_top5))
        dict1[str(i)] = {"sentence": original_text,
                         "token_hit1": is_top1,
                         "token_hit5": is_top5,
                         "token_hit5_unorder": top5_unorder,
                         "rel_token_hit5_unorder": np.sum(top5_unorder) / len(top5_unorder),
                         "rel_token_hit1": np.sum(is_top1) / len(is_top1),
                         "rel_token_hit5": np.sum(is_top5) / len(is_top5),
                         "candidates_checked": 1,
                         "tokens": original_ids,
                         "computation_time": time.time() - t1
                    }
        #print(dict1)
        #statistics["stP1"][k] = d[key]["stencil_token_hit1"]
        #statistics["stP5"][k] = d[key]["stencil_token_hit5"]
        #statistics["stp5_unorder"][k] = d[key]["stencil_token_hit5_unorder"]
