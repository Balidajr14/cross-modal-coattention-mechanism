#!/bin/bash
#SBATCH --job-name=training_attention
#SBATCH --mem=30G
#SBATCH -t 6:30:00
#SBATCH -c 32 
#SBATCH -N 1
#SBATCH -p gpu_radiomics 
#SBATCH -A radiomics_gpu
#SBATCH --gres=gpu:p100:1
#SBATCH -D /cluster/home/danglada/danglada/radcure-challenge/

USERNAME=$(whoami)
source /cluster/home/$USERNAME/.bashrc
conda activate radcure-challenge
python -m attention_dense_model.train\
          /cluster/projects/radiomics/Temp/michal/RADCURE-challenge/data/\
          /cluster/projects/radiomics/RADCURE-challenge/clinical_full.csv\
          --num_workers 32\
          --batch_size 16\
          --lr 0.0004380867619725786\
          --weight_decay 1.2065637774378936e-05\
          --dropout 0.27917710971118714\
          --max_epochs 200\
          --gpus 1\
          --lr_decay_factor 0.5\
          --lr_decay_milestones 60 100 140 180\
          --num_time_bins 40\
          --mtlr_smooth_factor 100\
          --num_checkpoints 1\
          --exp_name best_model\
          --hidden_dim 128\
          --feature_size 9\
          --num_heads 16\
          --block_shape 16\
          --early_stop_delta 0.0005\
          --early_stop_patience 20\
          --dropout_last 0.5715023112535688\
          --growth_rate 12\
          --pred_save_path ./attention_dense_model/predictions_dense.csv
