#!/bin/bash

{
set -e

# set training mode... MUST DOUBLE CHECK ALL THE BELOW
WIDTH=$1
SEED=0

N_EPOCHS_J=100   # number of epochs for joint
LRJ=0.001  # learning rate for joint
BATCH_SIZE=128
GPU=0
SAVE_EVERY=5
WEIGHT_DECAY=0.1

# set file names
HEAD=cluster_w${WIDTH}s${SEED}wd${WEIGHT_DECAY}
JOINTLY_LOG_DIR=logs/${HEAD}/joint

# first we train jointly
python train_jointly.py --log_dir "$JOINTLY_LOG_DIR" --seed $SEED --n_epochs "$N_EPOCHS_J" \
      --batch_size $BATCH_SIZE --lr $LRJ --resnet_width "$WIDTH" --gpu $GPU --save_every $SAVE_EVERY\
      --weight_decay $WEIGHT_DECAY
}
