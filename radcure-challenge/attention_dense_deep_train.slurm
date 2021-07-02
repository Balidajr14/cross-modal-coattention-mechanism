#!/bin/bash
#SBATCH --job-name=training_attention
#SBATCH --mem=30G
#SBATCH -t 6:30:00
#SBATCH -c 32 
#SBATCH -N 1
#SBATCH -p gpu_radiomics 
#SBATCH -A radiomics_gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -D /cluster/home/danglada/danglada/radcure-challenge/

USERNAME=$(whoami)
source /cluster/home/$USERNAME/.bashrc
conda activate radcure-challenge
python -m attention_dense_deep_model.train\
          /cluster/projects/radiomics/Temp/michal/RADCURE-challenge/data/\
          /cluster/projects/radiomics/RADCURE-challenge/clinical_full.csv\
          --num_workers 32\
          --batch_size 24\
          --lr 0.0001706653206390032\
          --weight_decay 1.20630632411806e-05\
          --dropout 0.17341169265081022\
          --max_epochs 200\
          --gpus 1\
          --lr_decay_factor 0.5\
          --lr_decay_milestones 60 100 140 180\
          --dropout_emr 0.2491808750558831\
          --num_time_bins 40\
          --mtlr_smooth_factor 10\
          --num_checkpoints 1\
          --exp_name attention_model_dense_deep_best\
          --hidden_dim 128\
          --feature_size 9\
          --num_heads 8\
          --block_shape 32\
          --early_stop_delta 0.0005\
          --early_stop_patience 20\
          --dropout_last 0.5304922773250583\
          --growth_rate 12\
          --pred_save_path ./attention_dense_deep_model/predictions_dense_deep.csv
