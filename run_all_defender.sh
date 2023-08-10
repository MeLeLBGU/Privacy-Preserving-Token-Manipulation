#!/bin/bash

declare -a REMAP_TYPES=("conv")
declare -a DATASETS=("imdb" "sst2")
declare -a STENCIL=("3" "7" "11")
#declare -a DATASETS=("imdb")

for k in "${STENCIL[@]}"
do
    for i in "${REMAP_TYPES[@]}"
    do
        for j in "${DATASETS[@]}"
        do
            sbatch defender.sh $i $j $k
        done
    done
done



