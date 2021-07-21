#!/bin/bash

{
set -eu
LR='1e-3'
WEIGHT_DECAY='1e-6'
N_EPOCHS=100
SEED=0
GPU=0
ROBUST="--robust"  # for no robust replace with empty string or comment out
MODEL_PATH='logs/pgl_KMeans2_dro/modelw16s0wd1e-1epoch100.pth'
WIDTH=16
N_CLUSTERS=2

python pseudogroup_classifier.py --resnet_width $WIDTH --lr $LR \
  --model_path $MODEL_PATH \
  --weight_decay $WEIGHT_DECAY --n_epochs $N_EPOCHS --seed $SEED --gpu $GPU\
  $ROBUST  --use_pseudogrouplabels --cluster_model KMeans --n_clusters $N_CLUSTERS
}
