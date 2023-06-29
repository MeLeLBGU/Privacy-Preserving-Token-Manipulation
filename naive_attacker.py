import sys
sys.path.append("lm-scorer/")
import numpy as np
import torch
from scipy.special import softmax
import utils
import time
from typing import *
torch.nn.functional.softmax
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from transformers import PreTrainedTokenizer
from remap_base import RemapBase
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

    def reset(self) :
        self.removed_sentences = 0

    def nucleus_beam_search(self, candidates_tokens: List[str], prefix_probabilities: List[float]) -> Tuple[List[int],List[float]] :
        # return a list of the indices that should be removed from the candidate text
    
        # We assume that the candidate_texts are such that the adjacent cells in the candidate_texts list have the same prefix
        # Another assumption is that the prefix probabilies is such that prefix_probabilities[i] corresponds to the 
        # candidate_text[i] candidate_text[i+1] suffix..
        for i, candidates_text in enumerate(candidates_tokens):
            if i == len(candidates_tokens) - 1:
                continue
            # Verify that these two candidates are from the same text...
            if candidates_tokens[i][0:len(candidates_tokens[i])-1] != candidates_tokens[i+1][0:len(candidates_tokens[i+1])-1]:
                log.error("The two candidates are not corresponding to one another", candidates_tokens[i], candidates_tokens[i-1])
                exit(0)

        # before we do the nuclus beam search we need to transform the input ids to text
        candidates_texts = utils.get_text_from_input_ids(self.tokenizer, candidates_tokens)
        print("candidates tokens and corresponding text:")
        print(candidates_tokens,candidates_texts)
        total_probability = 0.0
        prob_candidates = [] # first item in tuple is the probability, and the second is the candidate index
        probabilities = []
        for i, candidate_text in enumerate(candidates_texts):
            if i % 2 == 0:
                prefix_probability = prefix_probabilities[int(i/2)]
                prob1, ids1, tokens1, logits1 = self.scorer.tokens_score(candidate_text) # should return the logit
                print(prob1, ids1, tokens1, logits1)

            else:
                prob2, ids2, tokens2, logits2 = self.scorer.tokens_score(candidate_text) # should return the logit

                # Now that we have the two id candidates, and their token score we will softmax it and multiply
                # it with the probability of the prefix
                # the last value of logits is the EoS, so we take 1 before that
                print(prob2, ids2, tokens2, logits2)
                probs = softmax([logits1[-2], logits2[-2]])
                for j, prob in enumerate(probs):
                    total_probability = total_probability + prob * prefix_probability
                    prob_candidates.append((prob * prefix_probability, (i-1) + j))
                    probabilities.append(prob * prefix_probability)
                # verify the probs..
                # if prob_candidates[i][0] + prob_candidates[i+1][0] != prefix_probabilities[(i-1)/2]:
                #     log.error("Sum of probabilities is not 1!", prefix_probabilities[(i-1)/2], prob_candidates[i][0],prob_candidates[i+1][0])
        
        if total_probability < self.percentage_target:
            return  # no need to rmove
        
        # will be helpful to sort the array
        prob_candidates.sort(key=mysort_criteria)
        
        indices_to_remove = []
        # Now we will remove the 3% least contributing candidates
        for i in range(len(prob_candidates)):
            candidate_prob, candidate_index = prob_candidates[i]
            total_probability = total_probability - candidate_prob
            print("Index, probability:", candidate_index, candidate_prob)

            if total_probability > self.percentage_target:
                indices_to_remove.append(candidate_index)
                print("Going to remove:", candidate_index, candidate_prob)
                print("Going to remove:", candidates_tokens[candidate_index], probabilities[candidate_index])
            else:
                break

        if indices_to_remove != []:
            print("removing these indices:", indices_to_remove)
            print("Before prefix_probabilities:",probabilities)
            print("Before candidates_tokens:",candidates_tokens)
            ## safe removal, we mark the 
            for index in sorted(indices_to_remove, reverse=True):
                probabilities.pop(index)
                candidates_tokens.pop(index)
            print("After prefix_probabilities:",probabilities)
            print("After candidates_tokens:",candidates_tokens)

        log.info("Probabilities:")
        for i in range(len(candidates_tokens)):
            log.info(probabilities[i], candidates_tokens[i])
        exit(1)
        return candidates_tokens, prefix_probabilities    
        

def create_possible_permutation(reverse_vocab, input_ids, permu_count, NBS : NucleusBeamSearch):
    # a = list(map(lambda x: list(x), product([0, 1], [4, 5]))) // init
    # Loop:
    # c = [x[0] + [x[1]] for x in list(product(a.copy(), [4, 5]))]

    # token_choices1 = get_token_choices(reverse_vocab, input_ids[0], permu_count)
    # token_choices2 = get_token_choices(reverse_vocab, input_ids[1], permu_count)
    prefix_probabilities = [1.0]
    permutations = list(map(lambda x: list(x), product()))# this is just init [[]]
    # use pqdm ot tqdm
    for i, token in enumerate(input_ids):
        token = int(token)
        if token == 101 or token == 102:
            continue
        print(input_ids)
        token_choices = get_token_choices(reverse_vocab, token, permu_count)
        print("create_possible_permutation:", token, token_choices, reverse_vocab[token_choices[1]], reverse_vocab[token_choices[0]])
        permutations = [x[0] + [x[1]] for x in list(product(permutations.copy(), token_choices))]
        permutations, prefix_probabilities = NBS.nucleus_beam_search(permutations, prefix_probabilities)

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
        if score < real_score:
            rank = rank + 1
    return rank


def main(tokenizer, device, train_input_ids, val_input_ids, remapper: RemapBase, permu_count):
    scorer = LMScorer.from_pretrained("gpt2", device = device, batch_size = 1)
    NBS = NucleusBeamSearch(scorer, 0.97, tokenizer)
    f = open("rank_of_attacker.txt", "w")
    f.close()
    for i, train_input_id in enumerate(train_input_ids):
        if i == 2:
            exit(1)
        permutations, probabilities = create_possible_permutation(remapper.get_reversed_map(), train_input_id, permu_count, NBS)
        continue
        for i, permu in enumerate(permutations):
            if permu == train_input_id:
                prob_of_real_text = probabilities[i]
                break

        rank = get_rank_of_text(probabilities, prob_of_real_text)
        str1 = "rank of guess: " + str(rank) + " out of: " + str(len(probabilities)) + " ratio: " + str(rank / len(probabilities)) + "\n"
        print(str1)
        f = open("rank_of_attacker.txt", "a")
        f.writelines(str1)
        f.close()
        NBS.reset()
        exit(1)