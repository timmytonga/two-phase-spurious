import os
import torch
import numpy as np
from torch.utils.data import Subset
from data.label_shift_utils import prepare_label_shift_data
from data.confounder_utils import prepare_confounder_data
from data.dro_dataset import DRODataset

root_dir = '/home/thiennguyen/research/datasets/'

dataset_attributes = {
    'CelebA': {
        'root_dir': 'celebA'
    },
    'CUB': {
        'root_dir': 'cub'
    },
    'CIFAR10': {
        'root_dir': 'CIFAR10/data'
    },
    'MultiNLI': {
        'root_dir': 'multinli'
    }
}

for dataset in dataset_attributes:
    dataset_attributes[dataset]['root_dir'] = os.path.join(root_dir, dataset_attributes[dataset]['root_dir'])

shift_types = ['confounder', 'label_shift_step']


def prepare_data(args, train, return_full_dataset=False):
    # Set root_dir to defaults if necessary
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    if args.shift_type=='confounder':
        return prepare_confounder_data(args, train, return_full_dataset)
    elif args.shift_type.startswith('label_shift'):
        assert not return_full_dataset
        return prepare_label_shift_data(args, train)


def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')


def setup_data(args, rootdir,
               data_constructor,  # dir that contains data
               target_name,  # we are classifying whether the input image is blond or not
               confounder_names,  # we aim to avoid learning spurious features... here it's the gender
               model_type,  # for meta info
               augment_data=False, fraction=1.0, splits=('train', 'val', 'test')):
    """
    Get the dataset and loader -- my custom version?
    """
    full_dataset = data_constructor(root_dir=rootdir,
                                    target_name=target_name,
                                    confounder_names=confounder_names,
                                    model_type=model_type,
                                    augment_data=augment_data)  # augment data adds random resized crop and random flip.

    subsets = full_dataset.get_splits(
        # basically return the Subsets object with the appropriate indices for train/val/test
        splits,  # also implements subsampling --> just remove random indices of the appropriate groups in train
        train_frac=fraction,  # fraction means how much of the train data to use --> randomly remove if less than 1
        subsample_to_minority=(args.sampling_method == 'subsample'))

    dro_subsets = [
        DRODataset(
            subsets[split],  # process each subset separately --> applying the transform parameter.
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
        for split in splits]

    train_data, val_data, test_data = dro_subsets
    train_loader = train_data.get_loader(train=True, reweight_groups=(args.sampling_method == 'reweight'),
                                         batch_size=args.batch_size)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, batch_size=args.batch_size)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, batch_size=args.batch_size)
    data = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
            'train_data': train_data, 'val_data': val_data, 'test_data': test_data}
    return data

