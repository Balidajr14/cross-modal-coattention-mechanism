#!/bin/bash
#SBATCH --job-name=extract_attention
#SBATCH --mem=10G
#SBATCH -t 12:00:00
#SBATCH -c 32 
#SBATCH -N 1
#SBATCH -p gpu_radiomics 
#SBATCH -A radiomics_gpu
#SBATCH --gres=gpu:1
#SBATCH -D /cluster/home/danglada/danglada/radcure-challenge/

USERNAME=$(whoami)
source /cluster/home/$USERNAME/.bashrc
conda activate radcure-challenge
python -m attention_model.extract_attention\
          /cluster/home/danglada/danglada/radcure-challenge/data/logs/attention_model_best/2394181/checkpoints/epoch=132-val_loss=1.60e+00-roc_auc=0.83.ckpt
          