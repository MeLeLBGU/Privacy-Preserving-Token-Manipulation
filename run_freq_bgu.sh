#!/bin/bash


# example: sbatch bgu_sbatch.sh window16-freq-high.pt freq-high 16 sst2
BENCHMARK="imdb"
for i in 16 64 128 256 512 1024 2048 4096
do
sbatch bgu_sbatch.sh window$i-freq-high.pt freq-high $i $BENCHMARK
done
## for all/half
sbatch bgu_sbatch.sh all-freq-high.pt freq-high all $BENCHMARK
sbatch bgu_sbatch.sh half-freq-high.pt freq-high half $BENCHMARK

for i in 16 64 128 256 512 1024 2048 4096
do
sbatch bgu_sbatch.sh window$i-freq-low.pt freq-low $i $BENCHMARK
done

sbatch bgu_sbatch.sh all-freq-low.pt freq-low all $BENCHMARK
sbatch bgu_sbatch.sh half-freq-low.pt freq-low half $BENCHMARK