#!/bin/bash

declare -a REMAP_TYPES=("freq-high" "freq-low" "random")
declare -a DATASETS=("imdb")
#declare -a DATASETS=("imdb")

for i in "${REMAP_TYPES[@]}"
do
    for j in "${DATASETS[@]}"
    do
        sbatch attacker.sh $i $j
    done
done



