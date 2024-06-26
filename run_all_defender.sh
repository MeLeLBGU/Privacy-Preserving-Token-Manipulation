#!/bin/bash

declare -a REMAP_TYPES=("conv")
declare -a REMAPS=("all" "validation")
declare -a DATASETS=("sst2")
declare -a STENCIL=("3" "5" "7" "9" "11")
#declare -a DATASETS=("imdb")
#python main.py --remap=$1 --save=$2 --remap_type=$3 --dataset=$4 

for m in "${REMAPS[@]}"
do
    for k in "${STENCIL[@]}"
    do
        for i in "${REMAP_TYPES[@]}"
        do
            for j in "${DATASETS[@]}"
            do
                save=$i$j$k$m
                sbatch defender.sh $m $save $i $j $k
            done
        done
    done
done



