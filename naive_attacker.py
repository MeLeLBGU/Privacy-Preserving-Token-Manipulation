import pickle
import sys
sys.path.append("lm-scorer/")
import numpy as np
import torch
from scipy.special import softmax
import utils
import json
import os
import copy
import time
from typing import *
torch.nn.functional.softmax
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from transformers import PreTrainedTokenizer, AutoTokenizer
from remap_base import RemapBase, RemapRandom, RemapFrequency
# from Attacker.lm_scorer.models.auto import AutoLMScorer as LMScorer
from itertools import product 
import logging as log

def mysort_criteria(e: Tuple):
    return e[0]

class NucleusBeamSearch():
    def __init__(self, scorer1: LMScorer,scorer2: LMScorer, percentage_target: float, tokenizer, permu_count):
        self.percentage_target = 0.9975
        self.scorer1 = scorer1
        self.scorer2 = scorer2
        self.tokenizer = tokenizer
        self.removed_sentences = 0
        self.remove_percentage = 0.0025
        self.removed_per_step = [0] #* 66 # 66 max length
        self.step = -1
        a_file = open("roberta_gpt_mapper.pkl", "rb")
        self.roberta_gpt_mapper = pickle.load(a_file)
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        a_file.close()
        self.time = time.time()
        self.dynamically = True
        self.max_percentage_target = percentage_target
        self.permu_count = permu_count

    def reset(self) :
        self.step = -1
        self.time = time.time()
        self.removed_per_step = [] # 66 max length
        self.percentage_target = 0.9975
        self.remove_percentage = 0.0025


    def nucleus_beam_search(self, candidates_tokens: List[str], prefix_probabilities: List[float]) -> Tuple[List[int],List[float]] :
        # return a list of the indices that should be removed from the candidate text
        self.step = self.step + 1
        # We assume that the candidate_texts are such that the adjacent cells in the candidate_texts list have the same prefix
        # Another assumption is that the prefix probabilies is such that prefix_probabilities[i] corresponds to the 
        # candidate_text[i] candidate_text[i+1] suffix..
        self.removed_per_step.append(0)
        # before we do the nuclus beam search we need to transform the input ids to text
        # candidates_texts = utils.get_text_from_input_ids(self.tokenizer, candidates_tokens, skip_special=True)
        
        # We are scoring the sentences with GPT2, however, the tokens were generated from
        # RoBERTa, therefore, remap the RoBERTa tokens to GPT-2 (there exists such mapper
        # except for some Korean/Chinese/Japanese characters)
        # 
        gpt_tokens = copy.deepcopy(candidates_tokens)
        for i, tokens in enumerate(candidates_tokens):
            for j, token in enumerate(tokens):
                gpt_tokens[i][j] = self.roberta_gpt_mapper[token]
        candidates_texts = utils.get_text_from_input_ids(self.gpt_tokenizer, gpt_tokens)#, skip_special=True)
        candidates_texts2 = utils.get_text_from_input_ids(self.tokenizer, candidates_tokens)#, skip_special=True)
        # Just make sure that the texts are the same
        if candidates_texts != candidates_texts2:
            print("Error in RoBERTa to GPT transition")
            print(candidates_texts, candidates_texts2)
            print(gpt_tokens, candidates_tokens)
            return None, None

        total_probability = 0.0 #1 - np.sum(prefix_probabilities)
        prob_candidates = [] # first item in tuple is the probability, and the second is the candidate index
        probabilities = []
        candidates_logits = []
        logits = []
        # for i, candidate_text in enumerate(candidates_texts):
        # Go through all the candidates tokens and apply the lm scorer on them
        # if len(gpt_tokens) < 16:
        #     for i, candidate_token in enumerate(gpt_tokens):
        #         # LM-scorer originally applys the tokenization itself, I changed their code so they can
        #         # accept the tokens in a string format
        #         if i % self.permu_count != self.permu_count - 1:
        #             candidate = ' '.join(map(str, candidate_token))
        #             # we batch it now
        #             prob1, ids1, tokens1, logits1 = self.scorer1.tokens_score(candidate) # should return the logit
        #             logits.append(logits1[-2])
        #         else:
        #             candidate = ' '.join(map(str, candidate_token))
        #             prob2, ids2, tokens2, logits2 = self.scorer1.tokens_score(candidate) # should return the logit
        #             # prob, ids, tokens, logits = self.scorer.tokens_score([candidate1, candidate2]) # should return the logit
        #             logits.append(logits2[-2])# -2 because -1 is the end of sentence token
        #             # prefix_probability = prefix_probabilities[int((i - self.permu_count + 1)/self.permu_count)]
        #             candidates_logits.append(logits)
        #             logits = []
        # else:
        candidates_batch = []
        for i, candidate_token in enumerate(gpt_tokens):
            candidate = ' '.join(map(str, candidate_token))
            candidates_batch.append(candidate)
        scores_batch = self.scorer2.tokens_score(candidates_batch)
        # unpack
        for i, score_tuple in enumerate(scores_batch):
            logits.append(score_tuple[3][-2])
            if i % self.permu_count == self.permu_count - 1:
                candidates_logits.append(logits)
                logits = []

        # Calculate the probability of the candidate tokens
        for i, logits_with_same_prefix in enumerate(candidates_logits):
            prefix_probability = prefix_probabilities[i]
            probs = softmax(logits_with_same_prefix)
            # print(prefix_permutations, probs)
            for j, prob in enumerate(probs):
                total_probability = total_probability + prob * prefix_probability
                prob_candidates.append((prob * prefix_probability, i*len(probs) + j))
                probabilities.append(prob * prefix_probability)
        
        
        # will be helpful to sort the array and then remove it
        prob_candidates.sort(key=mysort_criteria)
        indices_to_remove = []

        if self.step > 6: # only after we have atleast 2^4 candidates we will consider to remove them
            if len(probabilities) > 250: # if we are looking at too much candidates let's remove some of  as it might explode
                if self.percentage_target > self.max_percentage_target:
                    self.percentage_target = self.percentage_target - self.remove_percentage# lower threshold
                    # Next iteration we lower the percentage
                else:
                    print("Bad probability", len(probabilities), self.max_percentage_target,self.percentage_target,self.step)
                    return None, None

            for i in range(len(prob_candidates)):
                candidate_prob, candidate_index = prob_candidates[i]
                total_probability = total_probability - candidate_prob
                if total_probability > self.percentage_target:
                    indices_to_remove.append(candidate_index)
                    self.removed_per_step[self.step] = self.removed_per_step[self.step] + 1
                else:
                    break
        if indices_to_remove != []:
            ## safe removal, we mark the 
            for index in sorted(indices_to_remove, reverse=True):
                probabilities.pop(index)
                candidates_tokens.pop(index)
        return candidates_tokens, probabilities    
        

def create_possible_permutation(reverse_vocab, input_ids, permu_count, NBS : NucleusBeamSearch):
    # a = list(map(lambda x: list(x), product([0, 1], [4, 5]))) // init
    # Loop:
    # c = [x[0] + [x[1]] for x in list(product(a.copy(), [4, 5]))]

    prefix_probabilities = [1.0]
    permutations = list(map(lambda x: list(x), product()))# this is just init [[]]
    # use pqdm ot tqdm
    print("Total candidates to check:", len(input_ids))
    for i, token in enumerate(input_ids):
        token = int(token)
        if token == 0:
            return permutations, prefix_probabilities
        try:
            token_choices = get_token_choices(reverse_vocab, token)
        except:
            print("bad tokens")
            return None, None
        # print("create_possible_permutation:", token, token_choices)
        permutations = [x[0] + [x[1]] for x in list(product(permutations.copy(), token_choices))]
        permutations, prefix_probabilities = NBS.nucleus_beam_search(permutations, prefix_probabilities)
        if not permutations:
            return None, None

    return permutations, prefix_probabilities


def get_token_choices(reverse_vocab, token):
    token_choices = []
    for i in range(len(reverse_vocab[int(token)])):
        token_choices.append(reverse_vocab[int(token)][i])
    return token_choices


def get_rank_of_text(scores, real_score):
    rank = 0
    for i, score in enumerate(scores):
        if score <= real_score:
            rank = rank + 1
    return rank


def get_token_hit(permutations, real_permutation, topk=5):
    token_hit_counter = [0] * topk
    token_length = len(real_permutation)
    topk = min(topk, len(permutations))
    top5_tokens = []
    for i in range(topk):
        proba, tokens = permutations[i]
        top5_tokens.append(tokens)
        for j in range(token_length):
            if real_permutation[j] == tokens[j]:
                token_hit_counter[i] = token_hit_counter[i] + 1
    return token_hit_counter, top5_tokens


def main(tokenizer, device, mapped_train_input_ids, original_train_input_ids, remapper: RemapBase, permu_count, attacker_file):
    scorer1 = LMScorer.from_pretrained("gpt2", device = device, batch_size = 1)
    scorer2 = LMScorer.from_pretrained("gpt2", device = device, batch_size = 16)
    reversed_mapper = remapper.get_reversed_map()
    NBS = NucleusBeamSearch(scorer1, scorer2, 0.0, tokenizer, len(reversed_mapper[15]))
    print(attacker_file)
    if os.path.exists(attacker_file):
        with open(attacker_file, "r") as f:
            dict1 = json.load(f)
            last_index = int(list(dict1.keys())[-1])
    else:
        f = open(attacker_file, "w")
        dict1 = {}
        f.close()
        last_index = -1
    # we assume that the train input id is clean for special tokens
    for i, train_input_id in enumerate(mapped_train_input_ids):
        print("##### Starting a new permutation ######")
        original_text = utils.get_text_from_input_ids(tokenizer, original_train_input_ids[i])
        print("text is: ", original_text)
        if i <= last_index:
            continue # we already added it to the json file
        t1 = time.time()
        permutations, probabilities = create_possible_permutation(reversed_mapper, train_input_id, permu_count, NBS)
        prob_of_real_text = 0.0
        if permutations: 

            # get the real sentence probability and remove it from the list
            for j, permu in enumerate(permutations):
                if permu == original_train_input_ids[i]:
                    prob_of_real_text = probabilities[j]
                    permu_of_real_text = original_train_input_ids[i]
                    permutations.pop(j)
                    probabilities.pop(j)
                    break
            if prob_of_real_text == 0.0: # real sentence wasn't considered
                permu_of_real_text = original_train_input_ids[i]
                
            prob_permu = list(map(lambda x, y:(x,y), probabilities, permutations))
            prob_permu.sort(key=mysort_criteria, reverse=True) # by probabilities
            # now we can find top@k
            token_hit, top5tokens = get_token_hit(prob_permu, permu_of_real_text)
            rank = get_rank_of_text(probabilities, prob_of_real_text)
            
            dict1[str(i)] = {"sentence": original_text,
                            "cumulative_probability": NBS.percentage_target,
                            "rank": len(probabilities) - rank + 1,
                            "candidates_checked": len(probabilities) + 1,
                            "real_sentence_probability": prob_of_real_text,
                            "token_hit": token_hit,
                            "tokens": original_train_input_ids[i],
                            "removed_per_step": NBS.removed_per_step,
                            "computation_time": time.time() - t1
                            }
        else:
            # we couldn't converge :(
            dict1[str(i)] = {"sentence": original_text,
                            "rank": -1,
                            "cumulative_probability": 0,
                            "candidates_checked": -1,
                            "real_sentence_probability": 0.0,
                            "token_hit": [i],
                            "removed_per_step": NBS.removed_per_step,
                            "computation_time": -1,
                            "tokens": original_train_input_ids[i]
                        }
        if i % 20 == 19:
            with open(attacker_file, "r+") as outfile:
                json.dump(dict1, outfile, indent=4)
        NBS.reset()
