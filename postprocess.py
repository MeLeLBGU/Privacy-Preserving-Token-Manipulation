import argparse
import sys

import json
import numpy as np
import time
from utils import *
import pickle
import matplotlib.pyplot as plt

def get_statistics(d):
    len_data = len(d)
    statistics = {"rank":[0] * len_data, "removed_per_step":[0] * 180, "computation_time": 0,
                   "candidates_checked": [0] * len_data,
                   "abs_edit_distance" : [0] * len_data,
                   "rel_edit_distance" : [0] * len_data,
                   "real_probability": [0] * len_data,
                   "mrr":[0] * len_data,
                   "P1":[0] * len_data,
                   "P5":[0] * len_data
                   }
    failed = 0
    # i saved it badly so need to convert it to string
    for k, key in enumerate(d):
        if d[key]["rank"] == -1:
            failed = failed + 1
            continue
        statistics["rank"][k] = d[key]["rank"]
        if d[key]["rank"] == 1:
            statistics["P1"][k] = 1
        if d[key]["rank"] < 5:
            statistics["P5"][k] = 1

        statistics["mrr"][k] = 1/float(d[key]["rank"]) #/ d[key]["candidates_checked"])
        statistics["computation_time"] = statistics["computation_time"] + d[key]["computation_time"]

        statistics["candidates_checked"][k] = d[key]["candidates_checked"]
        statistics["real_probability"][k] = d[key]["real_sentence_probability"]
        
        statistics["abs_edit_distance"][k] = len(d[key]["tokens"]) - d[key]["token_hit"][0] #/ float())) # normalize
        statistics["rel_edit_distance"][k] = (len(d[key]["tokens"]) - d[key]["token_hit"][0]) / len(d[key]["tokens"]) #d[key]["token_hit"][0] / len(d[key]["tokens"])
        for j, rps in enumerate(d[key]["removed_per_step"]):
            statistics["removed_per_step"][j] = statistics["removed_per_step"][j] + rps
    print("Failed:", failed)
    return statistics, len_data


def plot_statistics(statistics, len_data):
    print("Total computation time (min):", statistics["computation_time"] / 60, "for:", len_data, "sentences.")
    print("Average computation time (sec):", statistics["computation_time"] / len_data)
    print("Average rank:", np.average(statistics["rank"]))
    print("Average P@1:", np.average(statistics["P1"]))
    print("Average P@5:", np.average(statistics["P5"]))
    print("Average MRR:", np.average(statistics["mrr"]))
    print("Average absolute edit distance :", np.average(statistics["abs_edit_distance"]))
    print("Average relative edit distance :", np.average(statistics["rel_edit_distance"]))
    print("Average candidates checked:", np.average(statistics["candidates_checked"]))
    print("Average probability of real sentence:", np.average(statistics["real_probability"]))
    
    # print("Average % to99ken @1 hit:", np.average(statistics["token_hit"][:][0]))
    # print()
    # plt.hist(statistics["rank"], bins=[1, 6, 11, 201], edgecolor='black', linewidth=1.2)
    # plt.xlabel('# rank')
    # plt.ylabel('Occurences')
    # plt.title("Histogram of the rank of the real sentence")
    # plt.show()
    # plt.savefig(statistics["file"].split(".json")[0] + 'plot_rank.png')
    # plt.close()

    # ######## candidates ###########
    # plt.hist(statistics["candidates_checked"], bins=[1, 6, 51, 101], edgecolor='black', linewidth=1.2)
    # plt.xlabel('# rank')
    # plt.ylabel('Occurences')
    # plt.title("Histogram of the total candidates checked")
    # plt.show()
    # plt.savefig(statistics["file"].split(".json")[0] + 'plot_candidates.png')
    # plt.close()

    # ######## token hit ###########
    # plt.hist(statistics["token_hit"][0][:] ,edgecolor='black', linewidth=1.2)
    # plt.xlabel('# rank')
    # plt.ylabel('Occurences')
    # plt.title("Histogram of the token miss top@1")
    # plt.show()
    # plt.savefig(statistics["file"].split(".json")[0] + 'plot_tokenhit.png')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',  type = str,
                        dest = 'file', help = 'Save mode path.')
    parser.add_argument('--all', default = False, action="store_true",
                        dest = 'all', help = 'What base model to use')
    FILES = ["attacker_random_sst2.json", "attacker_random_imdb.json", "attacker_freq-high_sst2.json",
             "attacker_freq-high_imdb.json"]
    
    args = parser.parse_args()

    if args.all:
        for file in FILES:
            print("######################", file, "######################")
            with open(file, "r") as f:
                data = json.load(f)
                statistics, len_data = get_statistics(data)
            plot_statistics(statistics, len_data)
            print("")
    else:
        with open(args.file, "r") as f:
            data = json.load(f)
            statistics, len_data = get_statistics(data)
        statistics["file"] = args.file
        plot_statistics(statistics, len_data)
    
