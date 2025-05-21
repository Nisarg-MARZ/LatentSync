#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=2 --master_port=25679 -m scripts.train_unet \
    --unet_config_path "configs/unet/stage2.yaml"
