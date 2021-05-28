#!/bin/bash
{
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
SAVE_EVERY=20

# set file names
HEAD=w${WIDTH}s${SEED}
JOINTLY_LOG_DIR=logs/${HEAD}/joint


# first we train jointly
#python train_jointly.py --log_dir $JOINTLY_LOG_DIR --seed $SEED --n_epochs $N_EPOCHS_J --batch_size $BATCH_SIZE --lr $LRJ --resnet_width $WIDTH --gpu $GPU --save_every $SAVE_EVERY

#for epoch in $(seq $SAVE_EVERY $N_EPOCHS_J $SAVE_EVERY)
for epoch in 40 60 80 100
do
    MODEL_PATH=${JOINTLY_LOG_DIR}/model_${epoch}.pth
    RW_LOG_DIR=logs/${HEAD}/reweight${epoch}
    SS_LOG_DIR=logs/${HEAD}/subsample${epoch}
    TN_LOG_DIR=logs/${HEAD}/tau_norm${epoch}
    # then we train reweight
    python train_classifier.py --resnet_width $WIDTH --model_path $MODEL_PATH --lr $LRS --sampling_method 'reweight' --n_epochs $N_EPOCHS_RW --log_dir $RW_LOG_DIR --gpu $GPU --seed $SEED

    # train subsample
    python train_classifier.py --resnet_width $WIDTH --model_path $MODEL_PATH --lr $LRS --sampling_method 'subsample' --n_epochs $N_EPOCHS_SS --log_dir $SS_LOG_DIR --gpu $GPU --seed $SEED

    # tau_norm
    python tau_norm.py --resnet_width $WIDTH --model_path $MODEL_PATH --log_dir $TN_LOG_DIR --seed $SEED --max_tau 5.0 --step 51
    
done

exit $?
}
