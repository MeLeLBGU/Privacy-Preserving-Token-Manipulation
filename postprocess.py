import argparse
import sys

import json
import numpy as np
import time
from utils import *
import pickle
import matplotlib.pyplot as plt

def get_statistics(data):
    len_data = len(data)
    statistics = {"rank":[0] * len_data, "removed_per_step":[0] * 66, "computation_time": 0,
                   "candidates_checked": [0] * len_data, "token_hit" : [[0]*len_data for i in range(5)],
                   "real_probability": [0] * len_data
                   }
    failed = 0
    # i saved it badly so need to convert it to string
    for k, d in enumerate(data):
        d = dict(d)
        for i, key in enumerate(d):
            if d[key]["rank"] == -1:
                failed = failed + 1
                continue
            statistics["rank"][k] = d[key]["rank"]
            statistics["computation_time"] = statistics["computation_time"] + d[key]["computation_time"]
            statistics["candidates_checked"][k] = d[key]["candidates_checked"]
            statistics["real_probability"][k] = d[key]["real_sentence_probability"]
            
            for j, th in enumerate(d[key]["token_hit"]):
                statistics["token_hit"][j][k] = len(d[key]["tokens"]) - th #/ float())) # normalize

            for j, rps in enumerate(d[key]["removed_per_step"]):
                statistics["removed_per_step"][j] = statistics["removed_per_step"][j] + rps
    print("Failed:", failed)
    return statistics, len_data


def plot_statistics(statistics, len_data):
    print("Total computation time (min):", statistics["computation_time"] / 60, "for:", len_data, "sentences.")
    print("Average computation time (sec):", statistics["computation_time"] / len_data)
    print("Average rank:", np.average(statistics["rank"]))
    print("Average candidates checked:", np.average(statistics["candidates_checked"]))
    print("Average probability of real sentence:", np.average(statistics["real_probability"]))
    
    # print("Average % token @1 hit:", np.average(statistics["token_hit"][:][0]))
    print()
    plt.hist(statistics["rank"], bins=[1, 6, 11, 201], edgecolor='black', linewidth=1.2)
    plt.xlabel('# rank')
    plt.ylabel('Occurences')
    plt.title("Histogram of the rank of the real sentence")
    plt.show()
    plt.savefig(statistics["file"].split(".json")[0] + 'plot_rank.png')
    plt.close()

    ######## candidates ###########
    plt.hist(statistics["candidates_checked"], bins=[1, 6, 51, 101], edgecolor='black', linewidth=1.2)
    plt.xlabel('# rank')
    plt.ylabel('Occurences')
    plt.title("Histogram of the total candidates checked")
    plt.show()
    plt.savefig(statistics["file"].split(".json")[0] + 'plot_candidates.png')
    plt.close()

    ######## token hit ###########
    plt.hist(statistics["token_hit"][0][:] ,edgecolor='black', linewidth=1.2)
    plt.xlabel('# rank')
    plt.ylabel('Occurences')
    plt.title("Histogram of the token miss top@1")
    plt.show()
    plt.savefig(statistics["file"].split(".json")[0] + 'plot_tokenhit.png')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',  type = str,
                        dest = 'file', help = 'Save mode path.')

    args = parser.parse_args()
    with open(args.file, "r") as f:
        data = json.load(f)
    statistics, len_data = get_statistics(data)
    statistics["file"] = args.file
    plot_statistics(statistics, len_data)
    