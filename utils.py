import sys
import os
import torch
import numpy as np
import csv
from models import model_attributes
from data.data import dataset_attributes, shift_types
import logging


def get_logger(name: str,
               filename: str,
               l_format: str = "%(asctime)s:%(name)s:%(levelname)s: %(message)s",
               file_log_level: int = logging.INFO,
               output_to_console: bool = True,
               console_log_level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger configured to output to file and/or stdout
    :param name: the name of the logger -- usually __name__ for module name
    :param filename: the name for the log file for the logger to output to ex. "run2.log"
    :param l_format: a format string according to logging.Formatter (see Python docs for more info)
    :param file_log_level: logging level to log to the file
    :param output_to_console: whether we want this logger to output to stdout or not
    :param console_log_level: what level this logger should output to stdout
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(console_log_level)

    formatter = logging.Formatter(l_format)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if output_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class LoggerAdv(object):
    def __init__(self, fpath=None, mode='w', name=__name__):
        self.console = sys.stdout
        # self.file = None
        self.logger = None
        if fpath is not None:
            self.logger = get_logger(name, fpath)
            # self.file = open(fpath, mode)
    #
    # def __del__(self):
    #     self.close()
    #
    # def __enter__(self):
    #     pass
    #
    # def __exit__(self, *args):
    #     self.close()

    def write(self, msg, level="INFO"):
        assert level in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        self.console.write(msg)
        if self.logger is not None:
            if level == "INFO":
                self.logger.info(msg)
            elif level == "DEBUG":
                self.logger.debug(msg)
            elif level == "WARNING":
                self.logger.warning(msg)
            elif level == "ERROR":
                self.logger.error(msg)
            elif level == "CRITICAL":
                self.logger.critical(msg)

    def flush(self):
        pass
        # self.console.flush()
        # if self.file is not None:
        #     self.file.flush()
        #     os.fsync(self.file.fileno())

    def close(self):
        pass
        # self.console.close()
        # if self.file is not None:
        #     self.file.close()


class Logger(object):
    def __init__(self, fpath=None, mode='w', name=__name__):
        self.console = sys.stdout
        self.file = None
        # self.logger = None
        if fpath is not None:
            # self.logger = get_logger(name, fpath)
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode='w'):
        columns = ['epoch', 'batch']
        for idx in range(n_groups):
            columns.append(f'avg_loss_group:{idx}')
            columns.append(f'exp_avg_loss_group:{idx}')
            columns.append(f'avg_acc_group:{idx}')
            columns.append(f'processed_data_count_group:{idx}')
            columns.append(f'update_data_count_group:{idx}')
            columns.append(f'update_batch_count_group:{idx}')
        columns.append('avg_actual_loss')
        columns.append('avg_per_sample_loss')
        columns.append('avg_acc')
        columns.append('model_norm_sq')
        columns.append('reg_loss')

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("WARNING: USER HAVE SET SEED i.e. USES CUDA DETERMINISTIC MACHINE --> THIS SLOWS DOWN TRAINING DRAMATICALLY!")


def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')


def add_args(parser):
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--subsample_to_minority', action='store_true', default=False)
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--hinge', default=False, action='store_true')

    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)
    parser.add_argument('--resnet_width', type=int, default=None)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)


def make_group_acc_columns_for_csv_writer(n_groups, train=True):
    """
    Make appropriate columns for csv writer to keep track of stats.
    train is boolean indicating whether it's for a train_csv_writer or not -- has an additional loss column
    """
    total_acc_per_group = [f'total_acc:g{i}' for i in range(n_groups)]
    group_accs = [f'group{i}_acc' for i in range(n_groups)]
    group_margins = [f'group{i}_margin' for i in range(n_groups)]
    if train:
        return ['epoch', 'total_acc',  'split_acc', 'loss',
                'avg_margin'] + total_acc_per_group + group_accs + group_margins
    else:
        return ['epoch', 'total_acc',  'split_acc',
                'avg_margin'] + total_acc_per_group + group_accs + group_margins
