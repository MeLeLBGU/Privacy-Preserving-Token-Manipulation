import argparse
import sys
import os
import json
import numpy as np
import time
from utils import *
import pickle
import matplotlib.pyplot as plt


def merge_json_files(file_path):
    merged_contents = {}
    file_paths = [file_path]
    prefix = file_path.split('.json')[0]
    if 'imdb' in file_path :
        multiplier = 2500
    else:
        multiplier = 1200
    for i in range(1, 10):
        fname = prefix + f"_segment{i}0.json"
        if os.path.exists(fname):
            file_paths.append(fname)
            with open(fname, 'r', encoding='utf-8') as file_in:
                data = json.load(file_in)
                keys = list(data.keys())
                for old_key in keys:
                    data[str(int(old_key) + i * multiplier)] = data[old_key]
                    del data[old_key]
            merged_contents = {**data, **merged_contents}

    return merged_contents

def get_statistics(d):
    len_data = len(d)
    statistics = {"rank":[0] * len_data, "removed_per_step":[0] * 513, "computation_time": 0,
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

def get_statistics_knn(d):
    len_data = len(d)
    statistics = {"rank":[0] * len_data, "removed_per_step":[0] * 513, "computation_time": 0,
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
        statistics["P1"][k] = d[key]["rel_token_hit1"]
        statistics["P5"][k] = d[key]["rel_token_hit5"]
        statistics["computation_time"] = statistics["computation_time"] + d[key]["computation_time"]
        
        # statistics["abs_edit_distance"][k] = len(d[key]["tokens"]) - d[key]["token_hit"][0] #/ float())) # normalize
        statistics["rel_edit_distance"][k] = d[key]["rel_token_hit1"]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',  type = str,
                        dest = 'file', help = 'Save mode path.')
    parser.add_argument('--all', default = False, action="store_true",
                        dest = 'all', help = 'What base model to use')
    parser.add_argument('--segment', default = False, action="store_true",
                        dest = 'segment', help = 'Wether the json files are split to segments')
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
        if args.segment:
            data = merge_json_files(args.file)
        else:
            fname = args.file
            with open(fname, "r") as f:
                data = json.load(f)
        if "knn" in args.file:
            statistics, len_data = get_statistics_knn(data)
        else:
            statistics, len_data = get_statistics(data)
        statistics["file"] = args.file
        plot_statistics(statistics, len_data)
    
