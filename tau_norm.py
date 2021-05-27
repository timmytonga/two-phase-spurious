"""
Implements tau-normalization following [Kang' 20]: rescaling the weights of the last layer so that
    each column (representing the direction of each class in the feature space) has equal norm.
"""

import os
import argparse
from utils import set_seed, Logger, log_args
import torch
from data.celebA_dataset import CelebADataset
from data.dro_dataset import DRODataset
from data.data import log_data
from variable_width_resnet import resnet10vw
from train import run_epoch
import csv
import numpy as np

# first load the data and model
# then rescale
# then perform evaluation here also.


def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws

#######################################
#     Load args/data/model            #
#######################################
parser = argparse.ArgumentParser()
parser.add_argument('--resnet_width', type=int, default=None)
parser.add_argument('--model_path', type=str)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
# setup logging
mode = 'w'
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
# Record args
log_args(args, logger)
set_seed(args.seed)

device = f'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'  # no hard computation -- just use cpu
print('Using {} device'.format(device))

root_dir = '/home/thiennguyen/research/datasets/celebA/'  # dir that contains data
target_name = 'Blond_Hair'  # we are classifying whether the input image is blond or not
confounder_names = ['Male']  # we aim to avoid learning spurious features... here it's the gender
model_type = 'resnet10vw'
augment_data = False
fraction = 1.0
splits = ['train', 'val', 'test']

full_dataset = CelebADataset(root_dir=root_dir,
                             target_name=target_name,
                             confounder_names=confounder_names,
                             model_type=model_type,
                             augment_data=augment_data)  # augment data adds random resized crop and random flip.

subsets = full_dataset.get_splits(
    # basically return the Subsets object with the appropriate indices for train/val/test
    splits,  # also implements subsampling --> just remove random indices of the appropriate groups in train
    train_frac=fraction,  # fraction means how much of the train data to use --> randomly remove if less than 1
    subsample_to_minority=False)

dro_subsets = [
    DRODataset(
        subsets[split],  # process each subset separately --> applying the transform parameter.
        process_item_fn=None,
        n_groups=full_dataset.n_groups,
        n_classes=full_dataset.n_classes,
        group_str_fn=full_dataset.group_str)
    for split in splits]

train_data, val_data, test_data = dro_subsets
train_loader = train_data.get_loader(train=True, reweight_groups=False, batch_size=128)
val_loader = val_data.get_loader(train=False, reweight_groups=None, batch_size=128)
test_loader = test_data.get_loader(train=False, reweight_groups=None, batch_size=128)
data = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
        'train_data': train_data, 'val_data': val_data, 'test_data': test_data}
n_classes = 4  # since we are classifying the groups here
log_data(data, logger)
# load saved model

model = resnet10vw(args.resnet_width, num_classes=n_classes)
model.load_state_dict(torch.load(args.model_path))
model.to(device)

fc_weights = model.fc.weight
fc_bias = model.fc.bias

total_acc_per_group = [f'total_acc:g{i}' for i in range(4)]
valtest_columns = ['epoch', 'total_acc', 'group0_acc', 'group1_acc', 'group2_acc', 'group3_acc', 'split_acc',
                       'avg_margin', 'group0_margin',
                       'group1_margin', 'group2_margin', 'group3_margin'] + total_acc_per_group
val_path = open(os.path.join(args.log_dir, 'val.csv'), 'w')
val_writer = csv.DictWriter(val_path, fieldnames=valtest_columns)


def run_eval(x, reweighted_lastlayer, dataset, tau, writer):
    """
    Run the model with the reweighted last layer on the dataset
    """
    old_weight = x.fc.weight.clone()
    x.fc.weight = torch.nn.Parameter(reweighted_lastlayer)
    run_epoch(tau, x, device, None, dataset, None, writer, logger, is_training=False)
    x.fc.weight = torch.nn.Parameter(old_weight)


#######################################
#     Rescale last layer              #
#######################################
for p in np.linspace(0, 2, 21):
    ws = pnorm(fc_weights, p)
    run_eval(model, ws, val_loader, p, val_writer)

val_path.close()
