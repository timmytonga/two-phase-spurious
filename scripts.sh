#!/bin/bash

# This script is for saving commands
python train_jointly.py --log_dir w50_seed0_log --seed 0 --n_epochs 50 --bat
ch_size 128 --lr 0.0005 --resnet_width 50 --resume --resume_from 50
