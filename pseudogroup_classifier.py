"""
Implements pseudogroup (by clustering) classification:
    - First obtain pseudogroup by assigning examples to groups using clustering on features of labels
    and classification status (if it's correctly classified by the original classifier or not).
    - Then run dro or other group-label required methods on these pseudogroups
        - Note: validation is done on the real group labels (how are these determined?)
"""

import os
import argparse
from utils import set_seed, LoggerAdv as Logger, log_args
import torch
from data.celebA_dataset import CelebADataset
from data.dro_dataset import DRODataset
from data.data import log_data, setup_data
from variable_width_resnet import resnet10vw
from train import train


def main():
    sampling_methods = ['subsample', 'reweight']
    avail_losses = ['LDAM', 'CE']

    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet_width', type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None,
                        help="If this is not set then we are training from scratch")
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default='1e-6')
    parser.add_argument('--sampling_method', choices=sampling_methods, default=None)
    parser.add_argument('--n_epochs', type=int, default=25)
    parser.add_argument('--log_dir', type=str, default=None, help='Please set log dir to save results and checkpoints')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume_from', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0, help="Which GPU to use. To use CPU, set to -1")
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--loss_type', type=str, choices=avail_losses, default='CE',
                        help=f"Which loss to use. Avail {avail_losses}")
    parser.add_argument('--robust', action='store_true', default=False)
    parser.add_argument('--train_last_layer_only', action='store_true', default=False,
                        help="Turn this on to freeze all other layers and train only the last layer")
    parser.add_argument('--classifying_groups', action='store_true', default=False,
                        help="Turn this on for classifying groups instead of labels.")

    # extra argument for obtaining pseudogroups
    # use_pseudogrouplabel, cluster_model (kmeans, gmm?), n_clusters,
    parser.add_argument('--use_pseudogrouplabels', action='store_true', default=False)
    parser.add_argument('--cluster_model', type=str, default='KMeans')
    parser.add_argument('--n_clusters', type=int, default=2,
                        help="Number of cluster per group -- note we will have n_clusters*n_labels(*2)")
    # parse and check args
    args = parser.parse_args()
    check_args(args)
    # setup logging
    if not os.path.exists(args.log_dir):  # todo: auto log dir containing relevant info
        os.makedirs(args.log_dir)
    mode = 'w' if args.resume_from == 0 else 'a'
    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args into log
    log_args(args, logger)
    if args.seed is not None:
        set_seed(args.seed)

    # data related setup #
    # todo: setup so we load data from other datasets -- the info below can be stored as some metadata somewhere
    data = setup_data(args, data_constructor=CelebADataset,
                      rootdir='/home/thiennguyen/research/datasets/celebA/',  # dir that contains data
                      target_name='Blond_Hair',  # we are classifying whether the input image is blond or not
                      confounder_names=['Male'],  # we aim to avoid learning spurious features... here it's the gender
                      model_type='resnet10vw', augment_data=False, fraction=1.0, splits=('train', 'val', 'test'))
    # data is a dict containing dataset and loader for each split.
    log_data(data, logger)

    # get device
    device = f'cuda:{args.gpu}' if (args.gpu >= 0 and torch.cuda.is_available()) else 'cpu'
    logger.write('Using {} device'.format(device))
    # setup model #
    # Warning: classifying groups might not work if we are doing pseudogroups.
    n_classes = data['train_data'].n_groups if args.classifying_groups else data['train_data'].n_classes
    model = resnet10vw(args.resnet_width, num_classes=n_classes)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # now we obtain the pseudolabels if requested
    if args.use_pseudogrouplabels:
        pgl_dataset = obtain_pseudo_group_label_data(args, model, device, data['train_data'], data['train_loader'])
        pgl_train_loader = pgl_dataset.get_loader(train=True, reweight_groups=False, batch_size=args.batch_size)
        data['train_data'] = pgl_dataset
        data['train_loader'] = pgl_train_loader
        log_pgl_data(pgl_dataset, logger, data['train_data'].n_groups)

    process_model_layers(args, model)  # this freezes the layer if the appropriate argument is set.
    train(args, model, device, mode, data, logger, run_test=True)  # train model, save it, and log stats
    logger.write(f"Finished training!")  # mainly for timestamp in the log file


def check_args(args):
    assert args.resume_from == 0, "HAVE NOT IMPLEMENTED RESUME_FROM != 0...DO NOT SET!"
    if args.model_path is None:
        print("\tARGS INFO: Model_path is None --> Training from scratch.")
    if args.train_last_layer_only:
        print("\tARGS INFO: We are training ONLY the last layer.")
    else:
        print("\tARGS INFP: We are training the ENTIRE NETWORK")


def process_model_layers(args, model):
    # freeze everything except the last layer
    if args.train_last_layer_only:
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # make sure freezing really worked
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias


def log_pgl_data(pgl_dataset, logger, real_n_groups):
    logger.write("PGL DATA INFO:")
    group_to_pseudogroup = {i: {sg: 0 for sg in range(pgl_dataset.n_groups)} for i in
                            range(real_n_groups)}  # dict of {real_group: {pseudo_group_label: count}}
    pseudogroup_to_group = {sg: {i: 0 for i in range(real_n_groups)} for sg in range(pgl_dataset.n_groups)}

    for idx, sgl in idxs_to_subgroup_labels.items():
        group_to_pseudogroup[train_data.get_g(idx)][sgl] += 1
        pseudogroup_to_group[sgl][train_data.get_g(idx)] += 1
    logger.write(f"\tGroups to Pseudogroups count: \n\t{group_to_pseudogroup}")
    logger.write(f"\tPseudogroups to Groups count: \n\t{pseudogroup_to_group}")


def obtain_pseudo_group_label_data(args, model, device, dataset, dataloader):
    """
    Given a model, a data loader (usually for the train set)
        - First extract the features
        - Then
    Return a PseudoGroupLabelDataset
    """
    from feature_extractor import FeatureExtractor
    from data.pseudogrouplabels_dataset import PseudoGroupLabelsDataset
    from pseudo_group_labelers import PseudoGroupLabeler
    extractor = FeatureExtractor(model, device, layer_name='fc',
                                 get_layers_input=True)  # for ResNet10VW we are getting the last layer's input
    # the below runs through an epoch of the loader and save the feature in output_sets['activations']
    #   along with other meta data
    output_sets = extractor.extract_features(dataloader, show_progress=True)
    cluster_model = "KMeans"  # the constructor below will take care of the clustering
    pseudo_group_labeler = PseudoGroupLabeler(cluster_model, output_sets, dataset.n_classes)
    idxs_to_subgroup_labels, n_pseudogroups = pseudo_group_labeler.get_pseudo_group_labels(
        n_clusters=args.n_clusters, max_iter=args.max_iters)
    pgl_dataset = PseudoGroupLabelsDataset(dataset, idxs_to_subgroup_labels, n_pseudogroups)
    return pgl_dataset


if __name__ == '__main__':
    main()
