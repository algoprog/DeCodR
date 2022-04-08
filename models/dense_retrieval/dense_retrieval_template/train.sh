#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=64000
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:1
#SBATCH -t 02-00:00:00 
#SBATCH -o slurm-%j.out 

hostname
source /home/jkillingback/.bashrc
source /home/jkillingback/miniconda2/etc/profile.d/conda.sh
conda env list
conda activate neg-gen
conda env list
conda list

version="1"
nvidia-smi
train_cmd="\
python train.py
"

echo $train_cmd
eval $train_cmd
