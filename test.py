import sys
sys.path.append("..")
#import numpy as np
#import torch
#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
#from transformers import AutoTokenizer, AutoModel
#from datasets import load_dataset
import glob
import os
import lzma
import subprocess

fs = glob.glob("*.pkl")
for f in fs: #print(f)
    if "qnli" in f:
        x = f.replace(".pkl", ".pkl.xz")
        with open(x, "wb") as outf:
            subprocess.run(['lzma', '-z', '--threads=4', '-c', f],stdout=outf, check=True)

