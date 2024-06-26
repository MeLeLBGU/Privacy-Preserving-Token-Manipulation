import os
import glob
import os 
from glob import glob 
import pandas as pd
directory = '/home/reemha/Privacy-Playground'
txt_files = glob(os.path.join(directory, 'punc*conv*qnli*.txt'))
for txt_file in txt_files:
    acc = []
    #print(txt_file)
    with open(txt_file, "r") as f:
        txt = f.readlines()
        txts = txt[1:]
        for line in txts:
            # print(line)
            acc.append(float(line.split()[-1]))
        if (acc == []):
            continue
        base_path = os.path.basename(txt_file)
        window = base_path.split('_')[3]
        remap = base_path.split('_')[5]
        sigma = base_path.split('_')[6]
        if remap == "validation":
            print(f"Stencil_({window},{sigma}) {max(acc)}")


