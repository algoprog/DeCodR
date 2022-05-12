#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=64000
#SBATCH -p gpu-long
#SBATCH --gres=gpu:4
#SBATCH -t 04-00:00:00 
#SBATCH -o slurm-%j.out 


hostname
source /home/jkillingback_umass_edu/.bashrc
source /work/jkillingback_umass_edu/miniconda3/etc/profile.d/conda.sh
conda env list
conda activate irbase
conda env list


python -m torch.distributed.launch --nproc_per_node=4 -m tevatron_trainer\
  --output_dir "/work/jkillingback_umass_edu/checkpoints/tevatron/" \
  --model_name_or_path Luyu/co-condenser-marco \
  --do_train \
  --cache_dir "/work/jkillingback_umass_edu/cache/" \
  --save_steps 10000 \
  --fp16 \
  --per_device_train_batch_size 16 \
  --train_n_passages 9 \
  --learning_rate 5e-6 \
  --q_max_len 512 \
  --p_max_len 512 \
  --num_train_epochs 3 \
  --negatives_x_device \
  --grad_cache \
  --gc_q_chunk_size 8 \
  --gc_p_chunk_size 8