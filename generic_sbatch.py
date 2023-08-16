import argparse
import os
import sys

# SEGMENTS = [0]

def get_conv_cmd(s, d, r,sig, with_attacker):
    if with_attacker:
        cmd = f"python main.py --attacker=knn --stencil_sigma={sig} --remap={r} --remap_type=conv --dataset={d} --stencil_size={s} \n"
    else:
        cmd = f"python main.py --remap={r} --stencil_sigma={sig} --remap_type=conv --dataset={d} --save=conv_stencil{s}_{d}_{r}_{sig} --stencil_size={s} \n"
    return cmd

def get_other_cmd(rt, rc, d, r, seg, with_attacker):
    if with_attacker and seg != 0:
        cmd = f"python main.py --attacker=naive --segment={seg} --remap={r} --remap_count={rc} --remap_type={rt} --dataset={d} --frequency_window=half --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl \n"
    if with_attacker and seg == 0:
        cmd = f"python main.py --attacker=naive --remap={r} --remap_count={rc} --remap_type={rt} --dataset={d}  --frequency_window=half --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl \n"
    if not with_attacker:
        cmd = f"python main.py --remap={r} --remap_count={rc} --remap_type={rt} --dataset={d} --save={rt}_{rc}_{d}_{r} --frequency_window=half --frequency_path=/home/reemha/Privacy-Playground/wiki_freq_roberta.pkl \n"
    return cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--defender', default = "", type = str,
                        dest = 'defender', help = 'what type of remap. freq-high is mapping low to high (default: random)', choices=["all", "random", "freq-high","freq-low", "conv", "none"])
    parser.add_argument('--attacker', default = "", type = str,
                        dest = 'attacker', help = 'Initiate attacker. (default: naive)', choices=["naive", "knn"])
    args = parser.parse_args()
    exection_cmds = []
    jobs = []
    REMAP_COUNT = [2, 3]
    REMAPS = ["all"]#, "validation"]
    #REMAPS = ["all"]    
# DATASETS = ["sst2", "imdb"]
    DATASETS = ["imdb"] #"imdb"]
    # STENCIL_SIZE = [3, 5, 7, 9, 11]
    STENCIL_SIZE = [3, 7, 9, 11]
    STENCIL_SIGMA = [1, 0.8]
    SEGMENTS = [20, 30]
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

    if args.defender == "conv" or args.defender == "all" or args.attacker == "knn":
        for sig in STENCIL_SIGMA:
            for s in STENCIL_SIZE:
                for d in DATASETS:
                    for r in REMAPS:
                        exection_cmd = get_conv_cmd(s, d, r,sig, with_attacker)
                        jobs.append(attacker_job + "conv" + d)
                        exection_cmds.append(exection_cmd)
                        permu.append(f"{attacker_job}_conv_{d}_{s}_{r}_{sig}")
    
    if args.defender != "conv" and args.attacker != "knn":
        for seg in SEGMENTS:
            for rt in REMAP_TYPES:
                for rc in REMAP_COUNT:
                    for d in DATASETS:
                        for r in REMAPS:
                            if rc == 3 and rt != "random":
                                continue
                            exection_cmd = get_other_cmd(rt, rc, d, r, seg, with_attacker)
                            exection_cmds.append(exection_cmd)
                            jobs.append(attacker_job + rt + d)
                            permu.append(f"{attacker_job}{rc}_{d}_{r}")

    for i, cmd in enumerate(exection_cmds):
        jobname = jobs[i]
        sbatch_template = \
f"""#!/bin/bash
#SBATCH --partition main 
#SBATCH --time 6-10:30:00 
#SBATCH --job-name {jobname} 
#SBATCH --output {permu[i]}_%j.out
#SBATCH --gpus=1

module load cuda/11.2
{cmd}
"""
        fname = "sbatch_run.sh"
        with open("sbatch_run.sh", "w") as f:
            f.writelines(sbatch_template)
        print(sbatch_template)
        os.system(f"sbatch {fname}")
