import random

import numpy as np
from scipy.sparse import issparse
from tensorflow.keras.utils import Sequence

from meezer.knn import extract_knn


def generator_from_index(
    X, Y, index_path, k, batch_size, search_k=-1, precompute=True, verbose=1
):
    if k >= X.shape[0] - 1:
        raise Exception(
            '''k value greater than or equal to (num_rows - 1)
                        (k={}, rows={}). Lower k to a smaller
                        value.'''.format(
                k, X.shape[0]
            )
        )
    if batch_size > X.shape[0]:
        raise Exception(
            '''batch_size value larger than num_rows in dataset
                        (batch_size={}, rows={}). Lower batch_size to a
                        smaller value.'''.format(
                batch_size, X.shape[0]
            )
        )

    if verbose > 0:
        print('Extracting KNN from index')

    neighbour_matrix = extract_knn(
        X, index_path, k=k, search_k=search_k, verbose=verbose
    )

    knn_sequence = LabeledKnnTripletGenerator(
        X, Y, neighbour_matrix, batch_size=batch_size
    )

    return knn_sequence


class LabeledKnnTripletGenerator(Sequence):
    """TODO."""
    def __init__(self, X, Y, neighbour_matrix, batch_size=32):
        self.X, self.Y = X, Y
        self.neighbour_matrix = neighbour_matrix
        self.batch_size = batch_size
        self.hard_mode = 0

    def __len__(self):
        """TODO."""
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        """TODO."""
        batch_indices = range(
            idx * self.batch_size, min((idx + 1) * self.batch_size, self.X.shape[0])
        )

        label_batch = self.Y[batch_indices]
        triplet_batch = [
            self.knn_triplet_from_neighbour_list(
                row_index=row_index, neighbour_list=self.neighbour_matrix[row_index]
            )
            for row_index in batch_indices
        ]

        if issparse(self.X):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)

        return (
            tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]),
            tuple([np.array(label_batch), np.array(label_batch)]),
        )

    def knn_triplet_from_neighbour_list(self, row_index, neighbour_list):
        """
        Nate here: I'll be making most of my changes to this class, specifically this function here.

        Rather than just returning:
        "A random (unweighted) positive example chosen."

        I'll instead use the Annoy index to sometimes generate a negatively select an example from
        the neighbors list, and use the known labels to generate a positive example.

        """
        triplets = []

        # Take another example from that label as positive
        row_label = self.Y[row_index]

        all_labels_with_row_label = np.argwhere(self.Y == row_label).flatten()
        assert len(all_labels_with_row_label) > 1

        positive_ind = np.random.choice(all_labels_with_row_label)
        while positive_ind == row_index:
            positive_ind = np.random.choice(all_labels_with_row_label)

        if random.random() > self.hard_mode:  # TODO: allow this value to be set
            # Take a random data point that is not the label as the negative example
            negative_ind = np.random.randint(0, self.X.shape[0])
            while negative_ind in all_labels_with_row_label:
                negative_ind = np.random.randint(0, self.X.shape[0])
        else:
            # Take a random neighbour that is not a part of the label as negative
            potential_neighbors = np.setdiff1d(
                neighbour_list, all_labels_with_row_label
            )
            if len(potential_neighbors) > 0:
                negative_ind = np.random.choice(potential_neighbors)
            else:
                negative_ind = np.random.randint(0, self.X.shape[0])
                while negative_ind in all_labels_with_row_label:
                    negative_ind = np.random.randint(0, self.X.shape[0])

        triplets += [self.X[row_index], self.X[positive_ind], self.X[negative_ind]]

        return triplets
