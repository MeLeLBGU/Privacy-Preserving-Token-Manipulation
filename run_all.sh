#!/bin/bash

declare -a REMAP_TYPES=("freq-high" "random")
declare -a DATASETS=("sst2")
declare -a SEGMENTS=("40" "50")
#declare -a DATASETS=("imdb")
#python main.py --remap_type=$1 --dataset=$2 --attacker --remap=all --frequency_window=half

for k in "${SEGMENTS[@]}"
do
    for i in "${REMAP_TYPES[@]}"
    do
        for j in "${DATASETS[@]}"
        do
            sbatch attacker.sh $i $j $k
        done
    done
done



