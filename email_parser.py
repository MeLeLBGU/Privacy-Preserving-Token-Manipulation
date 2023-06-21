import argparse
import sys
import os
# import pandas as pd
import numpy as np
import pickle


def read_enron_mails(path):
    text = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            fname = os.path.join(root, name)
            with open(fname, "r") as f:
                text.append(f.readlines())
    return text


if __name__ == "__main__":
    data = read_enron_mails("/home/reemh/maildir/")
    a_file = open("enron-mail.pkl", "wb")
    pickle.dump(data, a_file)
    a_file.close()