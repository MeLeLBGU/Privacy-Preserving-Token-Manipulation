#!/bin/bash
#SBATCH --partition rtx6000,rtx4090,rtx3090,rtx2080 
#SBATCH --time 6-10:30:00 
#SBATCH --job-name attconvsst2 
#SBATCH --output new_envconv_stencil_roberta-base_9_sst2_all_1.0_True_%j.log
#SBATCH --gpus=1 
#SBATCH --qos=pintery
 ##SBATCH --mem=0
check_gpu_model() {
    local gpu_model="$1"
    if [[ "$gpu_model" == *"GTX 1080"* ]]; then
        echo "no"
    else
        echo "yes"
    fi
}
gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
source "$(conda info --base)/etc/profile.d/conda.sh"
env_name=$(check_gpu_model "$gpu_model")
if [[ "$env_name" == "yes" ]]; then
    conda activate privacy2
else
    conda activate Privacy
fi
echo $LD_LIBRARY_PATH
echo watt
python main.py --model=roberta-base --attacker=knn --stencil_stride=1 --stencil_sigma=1.0 --remap=all --remap_type=conv --dataset=sst2 --stencil_size=9  --punctuated  --save=punctuatedconv_stencil_roberta-base_9_sst2_all_1.0_1 

