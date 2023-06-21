#!/bin/bash

# example: sbatch window16-freq-high.pt freq-high 16 sst2
BENCHMARK="imdb"
for i in 16 64 128 256 512 1024 2048 4096
do
echo $i
cat window$i-freq-high.txt 
done
## for all/half
echo all
cat  all-freq-high.txt #freq-high all $BENCHMARK
echo half
cat half-freq-high.txt #freq-high half $BENCHMARK

for i in 16 64 128 256 512 1024 2048 4096
do
echo$i
cat window$i-freq-low.txt #freq-low $i $BENCHMARK
done
echo all
cat all-freq-low.txt #freq-low all $BENCHMARK
echo half
cat half-freq-low.txt # freq-low half $BENCHMARK