#!/bin/bash

python -m preprocess.data_processing_pipeline \
    --total_num_workers 96 \
    --per_gpu_num_workers 12 \
    --resolution 256 \
    --sync_conf_threshold 3 \
    --temp_dir /mnt/ml/training/Data/CelebV-HQ/latent_sync/processed/ \
    --input_dir /mnt/ml/training/Data/20s_sample/
