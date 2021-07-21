#!/bin/bash
{
set -eu
#for WIDTH in 50 
for WIDTH in 2 16 100
do
    ROOT_PATH=logs/w${WIDTH}s0/
    #for EPOCH in 25 50 75 100
    for EPOCH in 20 40 60 80
    do
        python tau_norm.py --resnet_width $WIDTH --model_path $ROOT_PATH/reweight$EPOCH/model_25.pth --log_dir $ROOT_PATH/RWTN/$EPOCH --max_tau 5.0 --step 51
        python tau_norm.py --resnet_width $WIDTH --model_path $ROOT_PATH/subsample$EPOCH/model_50.pth --log_dir $ROOT_PATH/SSTN/$EPOCH --max_tau 5.0 --step 51
    done
done
}
