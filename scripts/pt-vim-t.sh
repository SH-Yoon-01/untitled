#!/bin/bash
conda activate <your_env>
cd <path_to_Vim>/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model vim2_tiny --batch-size 128 --drop-path 0.0 --weight-decay 0.1 --num_workers 32 --data-path <path_to_IN1K_dataset> --output_dir ./output/vim2_tiny --no_amp
