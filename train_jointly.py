from data.data import log_data
from utils import set_seed, Logger, log_args
import argparse
import os
import torch
from variable_width_resnet import resnet10vw
from data.celebA_dataset import CelebADataset
from data.dro_dataset import DRODataset
from train import train


def main():
    parser = argparse.ArgumentParser()
    # args: log_dir, seed, lr, batchsize, width, n_epochs, weight_decay
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--resnet_width', type=int, default=None)
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_from', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=None)

    args = parser.parse_args()
    # setup logging
    mode = 'a'
    if not args.resume:
        args.resume_from = 0
        mode = 'w'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    logger.write(f'Train_jointly CelebA')
    log_args(args, logger)
    if args.seed is not None:
        set_seed(args.seed)

    # loading data
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.write('Using {} device'.format(device))
    root_dir = '/home/thiennguyen/research/datasets/celebA/'  # dir that contains data
    target_name = 'Blond_Hair'  # we are classifying whether the input image is blond or not
    confounder_names = ['Male']  # we aim to avoid learning spurious features... here it's the gender
    model_type = 'resnet10vw'  # what model we are using to process --> this is to determine the input size to
    # rescale the image
    augment_data = False

    fraction = 1.0
    splits = ['train', 'val', 'test']
    full_dataset = CelebADataset(root_dir=root_dir,
                                 target_name=target_name,
                                 confounder_names=confounder_names,
                                 model_type=model_type,
                                 # this string is to get the model's input size (for resizing) and input type (image
                                 # or precomputed)
                                 augment_data=augment_data)  # augment data adds random resized crop and random flip.

    subsets = full_dataset.get_splits(
        # basically return the Subsets object with the appropriate indices for train/val/test
        splits,  # also implements subsampling --> just remove random indices of the appropriate groups in train
        train_frac=fraction,  # fraction means how much of the train data to use --> randomly remove if less than 1
        subsample_to_minority=False)
    train_data, val_data, test_data = [
        DRODataset(
            subsets[split],  # process each subset separately --> applying the transform parameter.
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str) \
        for split in
        splits]  # group_str is a function that takes an int representing a group in the valid range and return the
    # group's name
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    data = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
            'train_data': train_data, 'val_data': val_data, 'test_data': test_data}
    n_classes = 4  # since we are classifying the groups here
    log_data(data, logger)
    
    logger.flush()
    # initialize model, loss, and optimizer

    model = resnet10vw(args.resnet_width, num_classes=n_classes)
    if args.resume:
        load_path = os.path.join(args.log_dir, f'model_{args.resume_from}.pth')
        assert os.path.exists(load_path), f"Model path {load_path} specified does not exist."
        model.load_state_dict(torch.load(load_path))
        logger.write(f"Loaded model from {load_path} successfully! Training...")
    model.to(device)

    train(args, model, device, mode, data, logger)  # train model, save it, and log stats


if __name__ == '__main__':
    main()
