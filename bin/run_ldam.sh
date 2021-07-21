#!/bin/bash
set -eu

N_EPOCH=25
GPU=1
LR=0.001

for WIDTH in 2 16 50 100
do 
    E='20 40 60 80 100'
    if [ $WIDTH = 50 ]
    then
        E='25 50 75 100'
    fi
    for epoch in $E
    do 
        ROOT=logs/w${WIDTH}s0/
        MODEL_PATH=$ROOT/joint/model_$epoch.pth
        LOG_DIR=$ROOT/ldam/$epoch
        python train_classifier.py --resnet_width $WIDTH --model_path $MODEL_PATH --lr $LR --n_epochs $N_EPOCH --log_dir $LOG_DIR --gpu $GPU --loss_type LDAM
    done
done
