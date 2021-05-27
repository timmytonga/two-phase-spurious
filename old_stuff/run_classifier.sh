#!/bin/bash

WIDTH=16
EPOCH=100
MODEL_PATH=./w${WIDTH}_seed0_log/model_${EPOCH}.pth
SAMPLING_METHOD='reweight'
RW_LOG_DIR=w${WIDTH}n${EPOCH}_finetune_reweight_log
SS_LOG_DIR=w${WIDTH}n${EPOCH}_finetune_subsample_log

LR='5e-4'
N_EPOCHS=50
SEED=0


python train_classifier.py --resnet_width $WIDTH --model_path $MODEL_PATH --lr $LR --sampling_method $SAMPLING_METHOD --n_epochs $N_EPOCHS --log_dir $RW_LOG_DIR --seed $SEED
python train_classifier.py --resnet_width $WIDTH --model_path $MODEL_PATH --lr $LR --sampling_method 'subsample' --n_epochs 100 --log_dir $SS_LOG_DIR --seed $SEED
