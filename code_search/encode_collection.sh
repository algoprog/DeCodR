#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=64000
#SBATCH -p gpu-long
#SBATCH --gres=gpu:1
#SBATCH -t 04-00:00:00 
#SBATCH -o slurm-%j.out 


hostname
source /home/jkillingback_umass_edu/.bashrc
source /work/jkillingback_umass_edu/miniconda3/etc/profile.d/conda.sh
conda env list
conda activate irbase
conda env list

python encode_collection.py
