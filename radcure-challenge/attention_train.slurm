#!/bin/bash
#SBATCH --job-name=training_attention
#SBATCH --mem=30G
#SBATCH -t 6:30:00
#SBATCH -c 32 
#SBATCH -N 1
#SBATCH -p gpu_radiomics 
#SBATCH -A radiomics_gpu
#SBATCH --gres=gpu:1
#SBATCH -D /cluster/home/danglada/danglada/radcure-challenge/

USERNAME=$(whoami)
source /cluster/home/$USERNAME/.bashrc
conda activate radcure-challenge
python -m attention_model.train\
          /cluster/projects/radiomics/Temp/michal/RADCURE-challenge/data/\
          /cluster/projects/radiomics/RADCURE-challenge/clinical_full.csv\
          --num_workers 32\
          --batch_size 32\
          --lr 0.0002141\
          --weight_decay 0.0009414\
          --dropout 0.08455\
          --max_epochs 200\
          --gpus 1\
          --growth_rate 32\
          --lr_decay_factor .5\
          --lr_decay_milestones 60 100 140 180\
          --num_time_bins 40\
          --mtlr_smooth_factor 1000\
          --num_checkpoints 1\
          --exp_name attention_model_best\
          --hidden_dim 256\
          --embedding_size 128\
          --feature_size 484\
          --num_heads 2\
          --block_shape 8\
          --early_stop_delta 0.0005\
          --early_stop_patience 50\
          --dropout_last 0.2\
          --pred_save_path ./attention_model/predictions_basic.csv
