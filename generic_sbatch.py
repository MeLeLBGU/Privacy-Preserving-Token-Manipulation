import argparse
import os
import sys
import subprocess

def get_conv_cmd(s, d, r, sig, stride, m, with_attacker, p, skip):
    save = f"--save=conv_stencil_{m}_{s}_{d}_{r}_{sig}_{stride}"
    if with_attacker:
        cmd = f"python main.py --model={m} --attacker=knn --stencil_stride={stride} --stencil_sigma={sig} --remap={r} --remap_type=conv --dataset={d} --stencil_size={s} "
    else:
        cmd = f"python main.py --model={m} --remap={r} --stencil_stride={stride} --stencil_sigma={sig} --remap_type=conv --dataset={d}  --stencil_size={s} "
    if p:
        save = f" --save=punctuatedconv_stencil_{m}_{s}_{d}_{r}_{sig}_{stride}"
        cmd = cmd + " --punctuated"
    if skip:
        save = save + "_skip"
        cmd = cmd + " --skip_stop_words"
    cmd = cmd + " " + save + " \n"
    return cmd

def get_conv_cmd_flan(s, d, r, sig, stride, m, with_attacker, p, skip):
    cmd = f"python t5_flan_mmlu.py --stencil_stride={stride} --stencil_sigma={sig} --remap_type=conv --stencil_size={s} "
    if p:
        cmd = cmd + " --punctuated"
    if skip:
        cmd = cmd + " --skip_stop_words"
    cmd = cmd + " " + " \n"
    return cmd

def get_noise_cmd(d, r, eta, m, with_attacker, skip):
    save = f"--save=noise_{m}_eta{eta}_{d}_{r}"
    if with_attacker:
        cmd = f"python main.py --model={m} --attacker=knn --remap={r} --remap_type=noise --dataset={d} --eta={eta} "
    else:
        cmd = f"python main.py --model={m} --remap={r} --remap_type=noise --dataset={d} --eta={eta}  "
    if skip:
        save = save + "_skip"
        cmd = cmd + " --skip_stop_words "
    cmd = cmd + " " + save + " \n"
    return cmd

def get_noise_cmd_flan(d, r, eta, m, with_attacker, skip):
    cmd = f"python t5_flan_mmlu.py   --remap_type=noise --eta={eta}  "
    if skip:
        cmd = cmd + "--skip_stop_words "
    cmd = cmd + " " + " \n"
    return cmd

def get_other_cmd(rt, rc, d, r, seg, m, with_attacker):
    if with_attacker and seg != 0:
        cmd = f"python main.py --model={m} --attacker=naive --segment={seg} --remap={r} --remap_count={rc} --remap_type={rt} --dataset={d} --frequency_window=half --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl \n"
    if with_attacker and seg == 0:
        cmd = f"python main.py --model={m} --attacker=naive --remap={r} --remap_count={rc} --remap_type={rt} --dataset={d}  --frequency_window=half --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl \n"
    if not with_attacker:
        cmd = f"python main.py --model={m} --remap={r} --remap_count={rc} --remap_type={rt} --dataset={d} --save={m}_{rt}_{rc}_{d}_{r} --frequency_window=half --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl \n"
    return cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--defender', default = "", type = str,
                        dest = 'defender', help = 'what type of remap. freq-high is mapping low to high (default: random)', choices=["all", "random", "noise", "freq-high","freq-low", "conv", "none"])
    parser.add_argument('--attacker', default = "", type = str,
                        dest = 'attacker', help = 'Initiate attacker. (default: naive)', choices=["naive", "knn"])
    parser.add_argument('--model', default = "t5-small", type = str,
                        dest = 'model', help = 'what model', choices=["roberta-base", "bert-base-uncased", "t5-base", "t5-small", "flan-t5-base"])
    parser.add_argument('--punctuated', default=False, action="store_true", dest="punctuated")
    parser.add_argument('--skip', default=False, action="store_true", dest="skip")
    parser.add_argument('--iter', default=0, type=int, dest="iter")

    args = parser.parse_args()
    exection_cmds = []
    jobs = []
    m = args.model
    REMAP_COUNT = [2]
    #REMAPS = ["all", "validation"]
    REMAPS = ["validation"]    
    REMAPS=["all", "validation"]
    DATASETS = ["imdb", "sst2"]#, "imdb"]
   # DATASETS = ["sst2", "imdb"]#, "sst2"]
    #DATASETS = ["qnli"] #, "imdb", "sst2"]
    DATASETS = ["sst2"]#, "imdb"]
    STENCIL_SIZE = [9]#,7,11,15]
    #STENCIL_SIZE = [15,5, 11,7, 9]

    #STENCIL_SIGMA = [ 0.6, 0.8, 1.0, 0.2]#,0.6,0.8, 0.2]
    STENCIL_SIGMA = [ 0.2,0.8, 0.6, 1.0]#,0.6,0.8, 0.2]
    #ETA = [100, 150]
    ETA = [ 150, 100]    
    #STENCIL_SIGMA = [0.2]
    SEGMENTS = [20, 30]
    STENCIL_STRIDE = [1]
    permu = []
    # SEGMENTS = [0]
    if args.attacker == "" and args.defender == "":
        exit(1)
    attacker_job = "def"
    with_attacker = False
    if args.attacker != "":
        with_attacker = True
        attacker_job = "att"
        REMAPS = ["all"]
    if args.defender != "conv":
        if args.defender == "all" or args.defender == "":
            REMAP_TYPES = ["freq-high", "freq-low", "random"]
        elif args.defender != "conv":
            REMAP_TYPES = [args.defender]

    if args.defender == "conv":
        for stride in STENCIL_STRIDE:
            for sig in STENCIL_SIGMA:
                for s in STENCIL_SIZE:
                    for d in DATASETS:
                        for r in REMAPS:
                            if "flan" in args.model:
                                exection_cmd = get_conv_cmd_flan(s, d, r,sig,stride,m, with_attacker, args.punctuated, args.skip)
                            else:
                                exection_cmd = get_conv_cmd(s, d, r,sig,stride,m, with_attacker, args.punctuated, args.skip)
                            jobs.append(attacker_job + "conv" + d)
                            exection_cmds.append(exection_cmd)
                            if args.skip:
                                permu.append(f"conv_stencil_{args.model}_{s}_{d}_{r}_{sig}_{args.punctuated}_skip")
                            else:
                                permu.append(f"conv_stencil_{args.model}_{s}_{d}_{r}_{sig}_{args.punctuated}")
    
    if args.defender == "noise":
        for eta in ETA:
            for d in DATASETS:
                for r in REMAPS:
                    if "flan" in args.model:
                        exection_cmd = get_noise_cmd_flan(d, r, eta,m, with_attacker,args.skip)
                    else:
                        exection_cmd = get_noise_cmd(d, r, eta,m, with_attacker,args.skip)
                    jobs.append(attacker_job + "noise" + d)
                    exection_cmds.append(exection_cmd)
                    permu.append(f"{attacker_job}_noise_{d}_{r}_{eta}")
    
    if args.defender != "noise" and args.defender != "conv" and args.attacker != "knn":
        for seg in SEGMENTS:
            for rt in REMAP_TYPES:
                for rc in REMAP_COUNT:
                    for d in DATASETS:
                        for r in REMAPS:
                            if rc == 3 and rt != "random":
                                continue
                            exection_cmd = get_other_cmd(rt, rc, d, r, seg,m, with_attacker)
                            exection_cmds.append(exection_cmd)
                            jobs.append(attacker_job + rt + d)
                            permu.append(f"{attacker_job}{rc}_{d}_{r}")
    jobs_ids = []
    if args.model == "t5-base":
        PARTITION = "rtx6000,rtx4090,rtx3090"
    else:
        PARTITION = "rtx6000,rtx4090,rtx3090,rtx2080"
    for i, cmd in enumerate(exection_cmds):
        jobname = jobs[i]
        sbatch_template = \
f"""#!/bin/bash
#SBATCH --partition {PARTITION} 
#SBATCH --time 6-10:30:00 
#SBATCH --job-name {jobname} 
#SBATCH --output new_env{permu[i]}_%j.log
#SBATCH --gpus=1 
#SBATCH --qos=pintery
 ##SBATCH --mem=0
check_gpu_model() {{
    local gpu_model=\"$1\"
    if [[ \"$gpu_model\" == *\"GTX 1080\"* ]]; then
        echo "no"
    else
        echo "yes"
    fi
}}
gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
source \"$(conda info --base)/etc/profile.d/conda.sh"
env_name=$(check_gpu_model \"$gpu_model\")
if [[ "$env_name" == "yes" ]]; then
    conda activate privacy2
else
    conda activate Privacy
fi
echo $LD_LIBRARY_PATH
echo watt
{cmd}
"""
        fname = "sbatch_run.sh"
        with open("sbatch_run.sh", "w") as f:
            f.writelines(sbatch_template)
        print(sbatch_template)
        result = subprocess.run(['sbatch', "sbatch_run.sh"], capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted as job {job_id}")
        ####################### RUN ONE AFTER ANOTHER
        if args.iter >= 1:
            for j in args.iter:
                sbatch_template = \
                    f"""#!/bin/bash
                    #SBATCH --partition main 
                    #SBATCH --time 6-10:30:00 
                    #SBATCH --job-name {jobname} 
                    #SBATCH --output {permu[i]}_%j.log
                    #SBATCH --gpus=1
                    #SBACT --dependency=afterany:{job_id}

                    module load cuda/11.2
                    {cmd}
                    """
                with open("sbatch_run.sh", "w") as f:
                    f.writelines(sbatch_template)
                result = subprocess.run(['sbatch', "sbatch_run.sh"], capture_output=True, text=True)
                job_prev = job_id
                job_id = result.stdout.strip().split()[-1]
                print(f"Submitted as job {job_id}, dependt on {job_prev}")
                
