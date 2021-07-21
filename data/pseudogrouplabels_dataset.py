import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class PseudoGroupLabelsDataset(Dataset):
    """
        Might not be extremely reliable yet... Just really basic implementation of returning the pseudo_group labels
        Input:
            dataset: a dataset that implements __getitem__
    """

    def __init__(self, dataset, idxs_to_subgroup_labels, n_pseudogroups, process_item_fn=None):
        self.dataset = dataset
        self.idxs_to_subgroup_labels = idxs_to_subgroup_labels
        # pseudo groups count below
        self.n_groups = n_pseudogroups  # note that this is different than self.dataset.n_groups (real # of groups)
        self.n_classes = self.dataset.n_classes
        self.process_item = process_item_fn  # this might not be right
        self.group_str = self.dataset.group_str  # this is wrong!!

        self._group_array = torch.tensor([v for _, v in idxs_to_subgroup_labels.items()])
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1) == self._group_array).sum(1).float()

    def __getitem__(self, idx):
        return self.dataset.get_x(idx), self.dataset.get_y(idx), self.idxs_to_subgroup_labels[idx], idx

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self.dataset.class_counts()

    def input_size(self):
        for x, y, g in self:
            return x.size()

    def get_loader(self, train, reweight_groups, **kwargs):
        if not train:  # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups:  # Training but not reweighting
            shuffle = True
            sampler = None
        else:  # Training and reweighting
            # # When the --robust flag is not set, reweighting changes the loss function
            # # from the normal ERM (average loss over each training example)
            # # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # # When the --robust flag is set, reweighting does not change the loss function
            # # since the minibatch is only used for mean gradient estimation for each group separately
            # group_weights = len(self) / self._group_counts
            # weights = group_weights[self._group_array]
            #
            # # Replacement needs to be set to True, otherwise we'll run out of minority samples
            # sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            # shuffle = False
            raise NotImplementedError("Haven't implemented reweighting yet for pseudogroup labels yet!!! Whoops")

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader
