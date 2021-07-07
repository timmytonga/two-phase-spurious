#!/bin/bash
set -eu

N_EPOCH=25
GPU=0
LR=0.0001

WEIGHT_DECAY=1e-2

for WIDTH in 2
# for WIDTH in 2 16 50 100
do 
    #E='20 40 60 80 100'
    E=20
    if [ $WIDTH = 50 ]
    then
        E='25 50 75 100'
    fi
    for epoch in $E
    do 
        ROOT=logs/w${WIDTH}s0/
        MODEL_PATH=$ROOT/joint/model_$epoch.pth
        LOG_DIR=$ROOT/gdro_l2$WEIGHT_DECAY/$epoch
        python train_classifier.py --resnet_width $WIDTH --model_path $MODEL_PATH --lr $LR --n_epochs $N_EPOCH --log_dir $LOG_DIR --gpu $GPU --weight_decay $WEIGHT_DECAY --robust --sampling_method reweight  # need to reweight for robust
    done
done
