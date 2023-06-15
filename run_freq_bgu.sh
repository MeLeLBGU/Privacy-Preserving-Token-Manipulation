#!/bin/bash


sbatch bgu_sbatch.sh window16-freq-high.pt freq-high 16
sbatch bgu_sbatch.sh window64-freq-high.pt freq-high 64
sbatch bgu_sbatch.sh window128-freq-high.pt freq-high 128
sbatch bgu_sbatch.sh window256-freq-high.pt freq-high 256
sbatch bgu_sbatch.sh window512-freq-high.pt freq-high 512
sbatch bgu_sbatch.sh window1024-freq-high.pt freq-high 1024
sbatch bgu_sbatch.sh window2048-freq-high.pt freq-high 2048
sbatch bgu_sbatch.sh window4096-freq-high.pt freq-high 4096

sbatch bgu_sbatch.sh window16-freq-low.pt freq-low 16
sbatch bgu_sbatch.sh window64-freq-low.pt freq-low 64
sbatch bgu_sbatch.sh window128-freq-low.pt freq-low 128
sbatch bgu_sbatch.sh window256-freq-low.pt freq-low 256
sbatch bgu_sbatch.sh window512-freq-low.pt freq-low 512
sbatch bgu_sbatch.sh window1024-freq-low.pt freq-low 1024
sbatch bgu_sbatch.sh window2048-freq-low.pt freq-low 2048
sbatch bgu_sbatch.sh window4096-freq-low.pt freq-low 4096
