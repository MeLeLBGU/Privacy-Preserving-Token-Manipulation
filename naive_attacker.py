import sys
sys.path.append("./Attacker/lm-scorer")
import numpy as np
import torch
import utils
from lm_scorer.models.auto import AutoLMScorer as LMScorer
# from Attacker.lm_scorer.models.auto import AutoLMScorer as LMScorer
from itertools import product


def create_possible_permutation(reverse_vocab, input_ids, permu_count=2):
    # a = list(map(lambda x: list(x), product([0, 1], [4, 5]))) // init
    # Loop:
    # c = [x[0] + [x[1]] for x in list(product(a.copy(), [4, 5]))]
    token_choices1 = get_token_choices(reverse_vocab, input_ids[0], permu_count)
    token_choices2 = get_token_choices(reverse_vocab, input_ids[1], permu_count)
    permutations = list(map(lambda x: list(x), product(token_choices1, token_choices2)))
    # use pqdm ot tqdm
    for i, token in enumerate(input_ids):
        if token == 0:
            break
        if i < 2:
            continue
        if i > 20: # too much
            return []
        token_choices = get_token_choices(reverse_vocab, input_ids[i], permu_count)

        permutations = [x[0] + [x[1]] for x in list(product(permutations.copy(), token_choices))]
    return permutations


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


def main(tokenizer, device, train_input_ids, val_input_ids, reverse_vocab, permu_count):
    scorer = LMScorer.from_pretrained("gpt2", device = device, batch_size = 1)
    f = open("rank_of_attacker.txt", "w")
    f.close()
    for i, train_input_id in enumerate(train_input_ids):
        permutations = create_possible_permutation(reverse_vocab, train_input_id, permu_count)
        print("Number permutations", len(permutations))
        if len(permutations) > np.power(2, 16) or permutations == []:
            continue
        guess_texts = utils.get_text_from_input_ids(tokenizer, permutations)
        real_text = utils.get_text_from_input_ids(tokenizer, train_input_id)
        score1 = get_scores(real_text, scorer, "mean")
        scores = get_scores(guess_texts, scorer, "mean")
        rank = get_rank_of_text(scores, score1)
        str1 = "rank of guess: " + str(rank) + " out of: " + str(len(scores)) + " ratio: " + str(rank / len(scores)) + "\n"
        print(str1)
        f = open("rank_of_attacker.txt", "a")
        f.writelines(str1)
        f.close()