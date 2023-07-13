#!/bin/bash


# example: sbatch bgu_sbatch.sh window16-freq-high.pt freq-high 16 sst2
BENCHMARK="imdb"
SAVEPATH="/home/reemha/Privacy-Playground/roberta_imdb"

sbatch bgu_sbatch.sh $SAVEPATH/random.pt random $i $BENCHMARK

for i in 16 256 512 1024 2048 4096
do
sbatch bgu_sbatch.sh $SAVEPATH/window$i-freq-high.pt freq-high $i $BENCHMARK
done
  
## for all/half
sbatch bgu_sbatch.sh $SAVEPATH/all-freq-high.pt freq-high all $BENCHMARK
sbatch bgu_sbatch.sh $SAVEPATH/half-freq-high.pt freq-high half $BENCHMARK

for i in 16 256 512 1024 2048 4096
do
sbatch bgu_sbatch.sh $SAVEPATH/window$i-freq-low.pt freq-low $i $BENCHMARK
done

sbatch bgu_sbatch.sh $SAVEPATH/all-freq-low.pt freq-low all $BENCHMARK
sbatch bgu_sbatch.sh $SAVEPATH/half-freq-low.pt freq-low half $BENCHMARK