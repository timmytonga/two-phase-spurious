import os
import argparse
from utils import set_seed, Logger, log_args
import torch
from data.celebA_dataset import CelebADataset
from data.dro_dataset import DRODataset
from data.data import log_data
from variable_width_resnet import resnet10vw
from train import train


def main():
    sampling_methods = ['subsample', 'reweight']
    avail_losses = ['LDAM', 'CE']

    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet_width', type=int, default=None)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--weight_decay', type=float, default='1e-6')
    parser.add_argument('--sampling_method', choices=sampling_methods, default=None)
    parser.add_argument('--n_epochs', type=int, default=25)
    parser.add_argument('--log_dir', type=str, default='./phase2_log')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume_from', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--loss_type', type=str, choices=avail_losses, default='CE')

    # '--dataset', 'CelebA', '-s', 'confounder', '-t', 'Blond_Hair', '-c', 'Male', '--log_dir', 'twophaselog_w32', '--seed', '0',
    # toparse = ['--resnet_width', '16',
    #            '--model_path', 'log_w16_seed0/last_model_50.pth',
    #            '--sampling_method', 'reweight']
    args = parser.parse_args()
    assert args.resume_from == 0, "HAVE NOT IMPLEMENTED RESUME_FROM != 0...DO NOT SET!"
    # setup logging
    mode = 'w'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)
    if args.seed is not None:
        set_seed(args.seed)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
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
    train_loader = train_data.get_loader(train=True, reweight_groups=(args.sampling_method == 'reweight'), batch_size=128)
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

    # freeze everything except the last layer
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # make sure freezing really worked
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    train(args, model, device, mode, data, logger)  # train model, save it, and log stats


if __name__ == '__main__':
    main()
