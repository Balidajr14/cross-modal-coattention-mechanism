#!/bin/bash

python search.py\
    "python -m attention_dense_deep_model.train\
          /cluster/projects/radiomics/Temp/michal/RADCURE-challenge/data/\
          /cluster/projects/radiomics/RADCURE-challenge/clinical_full.csv"\
    --config_file ../../data/hyperparams/hyperparams_deep.yaml\
    --hparams_save_path ../../data/hyperparams/\
    --num_samples 60\
    --max_concurrent_jobs 4\
    --source_bashrc\
    --conda_env radcure-challenge\
