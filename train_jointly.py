import numpy as np
from data.data import prepare_data, log_data
from utils import set_seed, Logger, log_args
import argparse
import os
import torch
from tqdm.auto import tqdm
from variable_width_resnet import resnet10vw
import csv
from pprint import pprint
from data.celebA_dataset import CelebADataset
from models import model_attributes
from data.dro_dataset import DRODataset

def main():
    parser = argparse.ArgumentParser()
    # args: log_dir, seed, lr, batchsize, width, n_epochs, weight_decay
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--resnet_width', type=int, default=None)
    parser.add_argument('--reweight_groups', action='store_true', default=False)

    args = parser.parse_args()
    # setup logging
    mode = 'w'  # todo: change this to resumeable
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    logger.write(f'Train_jointly CelebA')
    log_args(args, logger)
    set_seed(args.seed)

    # loading data

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
    # initialize model, loss, and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.write('Using {} device'.format(device))
    model = resnet10vw(args.resnet_width, num_classes=n_classes)
    model.to(device)

    train(args, model, device, mode, data, logger)  # train model, save it, and log stats


def train(args, model, device, mode, data, logger):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    train_path = open(os.path.join(args.log_dir, 'train.csv'), mode)
    val_path = open(os.path.join(args.log_dir, 'val.csv'), mode)
    test_path = open(os.path.join(args.log_dir, 'test.csv'), mode)

    train_columns = ['epoch', 'total_acc', 'group0_acc', 'group1_acc', 'group2_acc', 'group3_acc', 'split_acc', 'loss',
                     'avg_margin', 'group0_margin',
                     'group1_margin', 'group2_margin', 'group3_margin']

    valtest_columns = ['epoch', 'total_acc', 'group0_acc', 'group1_acc', 'group2_acc', 'group3_acc', 'split_acc',
                       'avg_margin', 'group0_margin', 'group1_margin', 'group2_margin', 'group3_margin']

    train_writer = csv.DictWriter(train_path, fieldnames=train_columns)
    train_writer.writeheader()
    val_writer = csv.DictWriter(val_path, fieldnames=valtest_columns)
    val_writer.writeheader()
    test_writer = csv.DictWriter(test_path, fieldnames=valtest_columns)
    test_writer.writeheader()

    train_loader, val_loader, test_loader = data['train_loader'], data['val_loader'], data['test_loader']
    for epoch in range(args.n_epochs):
        # train
        logger.write(f'Train epoch {epoch}')
        run_epoch(epoch + 1, model, device, optimizer, train_loader, criterion, train_writer, logger, is_training=True)

        # validate
        logger.write(f'Validate epoch {epoch}')
        run_epoch(epoch + 1, model, device, optimizer, val_loader, criterion, val_writer, logger, is_training=False)

    # Save model
    save_path = os.path.join(args.log_dir, f'last_model_{args.n_epochs}.pth')
    torch.save(model.state_dict(), save_path)

    # Save model
    save_path = os.path.join(args.log_dir, f'last_model_{args.n_epochs}.pth')
    torch.save(model.state_dict(), save_path)

    train_path.close()
    val_path.close()
    test_path.close()


def write_to_writer(writer, content):
    writer.writerow(content)
    pprint(content)


# refactor by making only 1 writer
def run_epoch(epoch, model, device, optimizer, loader, loss_computer, writer, logger, is_training):
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss, total_margin = 0, 0  # keep track of avg loss, margin in train
    l_correct, g_correct, total = 0, 0, 0  # for validation
    group_track = {f'g{i}': {'correct': 0, 'total': 0, 'margin': 0} for i in
                   range(4)}  # keeps track of counts of #correct, #total for each group g0-g4
    log_train_every = 200
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(tqdm(loader)):
            batch = tuple(t.to(device) for t in batch)
            x, y, g = batch
            outputs = model(x)

            if is_training:
                optimizer.zero_grad()
                loss = loss_computer(outputs, g)  # we are classifying groups
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if (batch_idx % log_train_every) == log_train_every - 1:  # print every 200 mini-batches
                    logger.write('[%d, %5d] loss: %.3f, avg_margin: %.3f' %
                                 (epoch, batch_idx + 1, running_loss / log_train_every, total_margin / total))
                    running_loss = 0.0

            # extra validation
            with torch.no_grad():
                # accuracies and margins
                total += g.size(0)  # total
                _, predicted = torch.max(outputs.data, 1)  # get predictions
                maskg = torch.zeros((len(g), 4), dtype=torch.bool, device=device)
                maskg[np.arange(len(g)), g] = 1
                margins = outputs[maskg] - torch.max(outputs * (~maskg), dim=1)[0]
                total_margin += margins.sum().item()
                for g_idx in range(4):
                    group_track[f'g{g_idx}']['correct'] += (
                        (predicted == g)[g == g_idx]).sum().item()  # correctly predict
                    group_track[f'g{g_idx}']['total'] += (
                            g == g_idx).sum().item()  # total number of group's instance encountered
                    group_track[f'g{g_idx}']['margin'] += margins[g == g_idx].sum().item()
                g_correct += (predicted == g).sum().item()
                l_correct += (predicted // 2 == y).sum().item()
        # write stats in dict and csv
        stats_dict = {'epoch': epoch, 'total_acc': f"{l_correct / total:.4f}", 'split_acc': f'{g_correct / total:.4f}',
                      'avg_margin': f"{total_margin / total:.4f}"}
        for g in range(4):
            stats_dict[f'group{g}_acc'] = f"{group_track[f'g{g}']['correct'] / group_track[f'g{g}']['total']:.4f}"
            stats_dict[f'group{g}_margin'] = f"{group_track[f'g{g}']['margin'] / group_track[f'g{g}']['total']:.4f}"
        if is_training:
            stats_dict['loss'] = f'{running_loss / log_train_every:.4f}'

        write_to_writer(writer, stats_dict)


if __name__ == '__main__':
    main()
