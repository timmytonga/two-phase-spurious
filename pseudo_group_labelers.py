"""
Here we implement the clustering part of the pipeline.
Given some inputs in the form of features, depending on the cluster type, return the appropriate pseudogroup labels
"""

import torch
import os
import numpy as np
from copy import copy  # copy the labels from the cluster
from sklearn.cluster import KMeans


class PseudoGroupLabeler:
    """
    This implements a bunch of different possible partition and cluster strategies to obtain pseudo-group-labels.
    """

    def __init__(self, cluster_model, output_sets, n_classes):
        """
        cluster_model: some cluster model. Here we use kmeans.
        output_sets: the dictionary as outlined in the feature_extractor -- contains
            'activations' (the features), the corresponding idx, labels, and model's prediction.
        n_classes: the number of classes we are dealing with.
        """
        self.cluster_model = cluster_model
        self.output_sets = output_sets
        self.n_classes = n_classes

    def get_pseudo_group_labels(self, n_clusters=2, max_iter=300, partition_correct_wrong=True):
        """
            Get pseudo_group_labels by clustering with cluster_model according to the label and additionally each point
                being correctly classified or not by the original model (output_sets['predicted'])
            Return an idxs_to_subgroup_labels dictionary and the
            TODO: Implement partition_correct_wrong=False
        """
        group_features_dict, group_idx_dict = self.partition_to_groups(partition_correct_wrong)
        return self.get_idxs_to_subgroup_labels(n_clusters, group_features_dict, group_idx_dict, max_iter,
                                                partition_correct_wrong)

    def get_idxs_to_subgroup_labels(self, n_clusters, group_features_dict, group_idx_dict, max_iter,
                                    partition_correct_wrong):
        """
        Given group_features_dict and group_idx_dict that contains the partition along with the corresponding
            idx/features partition each group into the number of clusters specified by n_clusters (the k in kMeans).
        Note the groups here can just be the labels or the labels and correctly/incorrectly classified partition.
        Return: idxs_to_subgroup_labels [dictionary {idx: label}], group_count [int]
            --> this can be passed into a pseudogrouplabel dataset for using with gDRO or other methods requiring group
                labels.
        """
        assert partition_correct_wrong, "Haven't been implemented yet!!!"
        cluster_assignments = {}
        for label in range(self.n_classes):
            for stat in ['correct', 'wrong']:
                cluster_model = self.cluster_model(n_clusters, max_iter=max_iter)
                cluster_model.fit(group_features_dict[f'class{label}_{stat}'])
                cluster_assignments[f'class{label}_{stat}'] = copy(cluster_model.labels_)
        # Now we map each idx for each point to its appropriate group label
        idxs_to_subgroup_labels = {}

        group_label_counter = 0  # so that the group labels for each groups get a unique number
        for k in cluster_assignments.keys():
            idx_array = group_idx_dict[k]
            assignment_array = cluster_assignments[k] + group_label_counter
            idxs_to_subgroup_labels.update({idx: assignment for idx, assignment in zip(idx_array, assignment_array)})
            group_label_counter += len(np.unique(cluster_assignments[k]))

        print("n_groups = ", group_label_counter)
        return idxs_to_subgroup_labels, group_label_counter

    def partition_to_groups(self, partition_correct_wrong=True):
        """
            Helper function for clustering --> arrange the features of a dataset into groups corresponding
                to labels and correctly classified or not. This partition is for separate clustering of different groups
            Return two dictionaries containing the partitions of features/idx that are matched according to the group.
        """
        if not partition_correct_wrong:
            raise NotImplementedError

        group_features_dict = {}  # this contains the features of the different labels/correctly-classified groups
        group_idx_dict = {}  # this contains the corresponding idx dict

        print(len(self.output_sets['activations']), self.output_sets['activations'][0].shape)
        x_feature = torch.cat(self.output_sets['activations'])
        y_array = torch.cat(self.output_sets['labels'])
        predicted = torch.cat(self.output_sets['predicted'])
        idxs = torch.cat(self.output_sets['idx'])

        # then we partition it into the combinations: label x [correctly, incorrectly classified]
        for label in range(self.n_classes):
            # first get the indices of the partition
            correct_select = (y_array == label) & (y_array == predicted)
            wrong_select = (y_array == label) & (y_array != predicted)
            # then we build the dictionaries per feature/idxs
            group_features_dict[f'class{label}_correct'] = x_feature[correct_select]
            group_idx_dict[f'class{label}_correct'] = idxs[correct_select].numpy()
            group_features_dict[f'class{label}_wrong'] = x_feature[wrong_select]
            group_idx_dict[f'class{label}_wrong'] = idxs[wrong_select].numpy()
        return group_features_dict, group_idx_dict
