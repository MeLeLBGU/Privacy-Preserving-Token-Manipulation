#!/bin/bash

declare -a REMAP_TYPES=("freq-high" "freq-low") # "random")
declare -a REMAP_COUNT=(2, 2, 2, 3)
declare -a DATASETS=("imdb")
declare -a REMAPS=("all" "validation")
#declare -a DATASETS=("imdb")
#--remap=$1 --save=$2 --remap_type=$3 --frequency_window=half --dataset=$4 --remap_count=$5 #--finetune
for i in "${REMAP_TYPES[@]}"
do
    for j in "${DATASETS[@]}"
    do
        for k in "${REMAPS[@]}"
            do
            save=$i$j$k
            sbatch bgu_sbatch.sh $k  $save $i $j 2
        done
    done
done

# i="random"
# for j in "${DATASETS[@]}"
# do
#     for k in "${REMAPS[@]}"
#         do
#         save=3$i$j$k
#         sbatch bgu_sbatch.sh $k  $save $i $j 3
#     done
# done



