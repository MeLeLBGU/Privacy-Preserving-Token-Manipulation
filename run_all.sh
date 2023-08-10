#!/bin/bash

declare -a REMAP_TYPES=("freq-high" "random")
declare -a DATASETS=("imdb")
declare -a SEGMENTS=("40" "50")
#declare -a DATASETS=("imdb")

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



