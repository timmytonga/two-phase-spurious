#!/bin/bash
set -eu
: $1, $2

# set training mode... MUST DOUBLE CHECK ALL THE BELOW
WIDTH=$1
SEED=0
N_EPOCHS_J=$2   # number of epochs for joint
N_EPOCHS_RW=25  # number of epochs for reweighting
N_EPOCHS_SS=50  # number of epochs for subsampling
LRJ=0.001  # learning rate for joint
LRS=0.001  # learning rate for subsample/reweighting
BATCH_SIZE=128
GPU=1
SAVE_EVERY=25

# set file names
HEAD=w${WIDTH}s${SEED}n${N_EPOCHS_J}
JOINTLY_LOG_DIR=logs/${HEAD}/joint
RW_LOG_DIR=logs/${HEAD}/reweight
SS_LOG_DIR=logs/${HEAD}/subsample

MODEL_PATH=${JOINTLY_LOG_DIR}/model_${N_EPOCHS_J}.pth

# first we train jointly
python train_jointly.py --log_dir $JOINTLY_LOG_DIR --seed $SEED --n_epochs $N_EPOCHS_J --batch_size $BATCH_SIZE --lr $LRJ --resnet_width $WIDTH --gpu $GPU --save_every $SAVE_EVERY

# then we train reweight
python train_classifier.py --resnet_width $WIDTH --model_path $MODEL_PATH --lr $LRS --sampling_method 'reweight' --n_epochs $N_EPOCHS_RW --log_dir $RW_LOG_DIR --seed $SEED --gpu $GPU

# finally train subsample
python train_classifier.py --resnet_width $WIDTH --model_path $MODEL_PATH --lr $LRS --sampling_method 'subsample' --n_epochs $N_EPOCHS_SS --log_dir $SS_LOG_DIR --seed $SEED --gpu $GPU

