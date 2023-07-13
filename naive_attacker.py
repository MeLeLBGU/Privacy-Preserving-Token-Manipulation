import pickle
import sys
sys.path.append("lm-scorer/")
import numpy as np
import torch
from scipy.special import softmax
import utils
import json
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
    def __init__(self, scorer: LMScorer, percentage_target: float, tokenizer):
        self.percentage_target = percentage_target
        self.scorer = scorer
        self.tokenizer = tokenizer
        self.removed_sentences = 0
        self.removed_per_step = [0] #* 66 # 66 max length
        self.step = -1
        a_file = open("roberta_gpt_mapper.pkl", "rb")
        self.roberta_gpt_mapper = pickle.load(a_file)
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        a_file.close()
        self.time = time.time()
        self.dynamically = True
        self.max_percentage_target = percentage_target

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
        
        # if self.step % 10 == 0:
        #     # print("Finished 2^10 candidates with time:", time.time() - self.time)
        #     print("Candidates to check", len(candidates_tokens))
        #     print("Removed:", np.sum(self.removed_per_step))

        if len(candidates_tokens) > 2:
            for i in range(0, len(candidates_tokens), 2):
                # Verify that these two candidates are from the same text...
                if candidates_tokens[i][0:-1] != candidates_tokens[i+1][0:-1]:
                    print("The two candidates are not corresponding to one another", candidates_tokens[i], candidates_tokens[i+1])
                    print(candidates_tokens)
                    # log.error("The two candidates are not corresponding to one another", candidates_tokens[i], candidates_tokens[i+1])
                    exit(0)

        # before we do the nuclus beam search we need to transform the input ids to text
        # candidates_texts = utils.get_text_from_input_ids(self.tokenizer, candidates_tokens, skip_special=True)
        
        # we map from RoBERTa to gpt
        gpt_tokens = copy.deepcopy(candidates_tokens)
        for i, tokens in enumerate(candidates_tokens):
            for j, token in enumerate(tokens):
                gpt_tokens[i][j] = self.roberta_gpt_mapper[token]
        candidates_texts = utils.get_text_from_input_ids(self.gpt_tokenizer, gpt_tokens)#, skip_special=True)
        candidates_texts2 = utils.get_text_from_input_ids(self.tokenizer, candidates_tokens)#, skip_special=True)
        if candidates_texts != candidates_texts2:
            print("Error in RoBERTa to GPT transition")
            print(candidates_texts, candidates_texts2)
            print(gpt_tokens, candidates_tokens)
            return None, None
#        print("text:", candidates_texts)

        total_probability = 0.0 #1 - np.sum(prefix_probabilities)
        prob_candidates = [] # first item in tuple is the probability, and the second is the candidate index
        probabilities = []
        # for i, candidate_text in enumerate(candidates_texts):
        for i, candidate_token in enumerate(gpt_tokens):
            # if len(gpt_tokens) > 2000 and i % 1000 == 0:
                # print("Iteration:", i, "out of:", len(gpt_tokens))
            if i % 2 == 0:
                prefix_probability = prefix_probabilities[int(i/2)]
                candidate = ' '.join(map(str, candidate_token))
                # we batch it now
                prob1, ids1, tokens1, logits1 = self.scorer.tokens_score(candidate) # should return the logit
            else:
                candidate = ' '.join(map(str, candidate_token))
                prob2, ids2, tokens2, logits2 = self.scorer.tokens_score(candidate) # should return the logit
                # prob, ids, tokens, logits = self.scorer.tokens_score([candidate1, candidate2]) # should return the logit

                # safe check the token lengths..
                if len(prob1) != len(prob2):
                    print("candidates texts:", candidates_texts[i-1], candidates_texts[i])
                    print("candidates tokens:", tokens1,tokens2)
                    print("candidates gpt tokens:", gpt_tokens[i-1], gpt_tokens[i])
                    print("candidates gpt text:", self.gpt_tokenizer.decode(gpt_tokens[i-1]), self.gpt_tokenizer.decode(gpt_tokens[i]))
                    print("candidates ids:", ids1,ids2)
                    print("candidates probs:", prob1,prob2)
                    print("The two candidates are not the same length", tokens1, tokens2)
                    return None, None
                # Now that we have the two id candidates, and their token score we will softmax it and multiply
                # it with the probability of the prefix
                # the last value of logits is the EoS, so we take 1 before that
                # print(prob2, ids2, tokens2, logits2)
                probs = softmax([logits1[-2], logits2[-2]])
                for j, prob in enumerate(probs):
                    total_probability = total_probability + prob * prefix_probability
                    prob_candidates.append((prob * prefix_probability, (i-1) + j))
                    probabilities.append(prob * prefix_probability)
                # verify the probs..
        
        # if total_probability < self.percentage_target:
        #     print("no more remov")
        #     return candidates_tokens, probabilities  # no need to rmove
        
        # will be helpful to sort the array
        prob_candidates.sort(key=mysort_criteria)
        
        indices_to_remove = []

        if self.step > 5: # only after we have atleast 2^6 candidates we will consider to remove them
            if len(probabilities) > 2000: # if we are looking at too much candidates let's remove some of em
                if self.percentage_target > self.max_percentage_target:
                    self.percentage_target = self.percentage_target - self.remove_percentage# lower threshold
                else:
                    print("Bad probability")
                    return None, None
                # print("allowing the removal of more candidates", self.step, self.percentage_target)
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

    # token_choices1 = get_token_choices(reverse_vocab, input_ids[0], permu_count)
    # token_choices2 = get_token_choices(reverse_vocab, input_ids[1], permu_count)
    prefix_probabilities = [1.0]
    permutations = list(map(lambda x: list(x), product()))# this is just init [[]]
    # use pqdm ot tqdm
    print("Total candidates to check:", len(input_ids))
    for i, token in enumerate(input_ids):
        token = int(token)
        token_choices = get_token_choices(reverse_vocab, token, permu_count)
        # print("create_possible_permutation:", token, token_choices)
        permutations = [x[0] + [x[1]] for x in list(product(permutations.copy(), token_choices))]
        permutations, prefix_probabilities = NBS.nucleus_beam_search(permutations, prefix_probabilities)
        if not permutations:
            return None, None

    return permutations, prefix_probabilities


def get_token_choices(reverse_vocab, token, permu_count=2):
    token_choices = []
    for i in range(permu_count):
        token_choices.append(reverse_vocab[int(token)][i])
    return token_choices


def get_scores(guess_texts, scorer, reduce_type="mean"):
    return scorer.sentence_score(guess_texts, reduce=reduce_type)


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
    scorer = LMScorer.from_pretrained("gpt2", device = device, batch_size = 1)
    NBS = NucleusBeamSearch(scorer, 0.0, tokenizer)
    print(attacker_file)
    f = open(attacker_file, "w")
    data = []
    topk = 5
    f.close()
    
    # we assume that the train input id is clean for special tokens
    for i, train_input_id in enumerate(mapped_train_input_ids):
        print("##### Starting a new permutation ######")
        original_text = utils.get_text_from_input_ids(tokenizer, original_train_input_ids[i])
        print("text is: ", original_text)
        t1 = time.time()
        permutations, probabilities = create_possible_permutation(remapper.get_reversed_map(), train_input_id, permu_count, NBS)
        prob_of_real_text = 0.001
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
            
            dict = {str(i): {"sentence": original_text,
                            "cumulative_probability": NBS.percentage_target,
                            "rank": len(probabilities) - rank + 1,
                            "candidates_checked": len(probabilities) + 1,
                            "real_sentence_probability": prob_of_real_text,
                            "token_hit": token_hit,
                            "removed_per_step": NBS.removed_per_step,
                            "computation_time": time.time() - t1
                            }}
            # print(dict)
        else:
            # we couldn't converge :(
            dict = {str(i): {"sentence": original_text,
                        "rank": -1,
                        "cumulative_probability": 0,
                        "candidates_checked": -1,
                        "real_sentence_probability": 0.0,
                        "token_hit": [-1],
                        "removed_per_step": NBS.removed_per_step,
                        "computation_time": -1
                        }} 
        data.append(dict)
        if i % 50 == 49:
            with open(attacker_file, "r+") as outfile:
                json.dump(data, outfile, indent=4)
        NBS.reset()
