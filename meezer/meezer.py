import json
import os
import platform
import shutil

import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import load_model, Model
from tqdm.auto import tqdm

from meezer.knn import build_annoy_index, extract_knn
from meezer.losses import (is_hinge,
                           is_multiclass,
                           semi_supervised_loss,
                           triplet_loss,
                           validate_sparse_labels)
from meezer.network import base_network, triplet_network
from meezer.triplet_generators import generator_from_index


class Meezer(BaseEstimator):
    """
    Meezer is a supervised Siamese network that uses a triplet loss function to seperate points from
    the same class and all other points. Over time, the pipeline introduces harder and harder
    negative examples to be used in the triplet loss using nearest-neighbors from an Annoy index
    generated after each epoch.

    Parameters
    -------------
    embedding_dims: int
        Number of dimensions in the embedding space (default 2)
    k: int
        The number of neighbors to retrieve for each point. Must be less than one minus the number
        of rows in the dataset (default 150)
    distance: str
        The loss function used to train the neural network. One of:
            * 'pn'
            * 'euclidean'
            * 'manhattan_pn'
            * 'manhattan'
            * 'chebyshev'
            * 'chebyshev_pn'
            * 'softmax_ratio_pn'
            * 'softmax_ratio'
            * 'cosine'
            * 'cosine_p'
        (default 'pn')
    batch_size: int
        The size of mini-batches used during gradient descent while training the neural network.
        Must be less than the number of rows in the dataset (default '128')
    epochs: int
        The maximum number of epochs to train the model for. Each epoch the network will see a
        triplet based on each data-point `sub_epochs` times. After each epoch, callbacks are reset
        and new hard negative examples are computed in an Annoy index (default 20)
    sub_epochs: int
        Number of epochs to train the model with a single set of hard negative examples for each of
        the epochs in `epochs` (default 10)
    margin: float
        The distance that is enforced between points by the triplet loss functions (default 1.)
    ntrees: int
        The number of random projections trees built by Annoy to approximate KNN. The more trees,
        the higher the memory usage, but the better the accuracy of results (default 50)
    search_k: int
        The maximum number of nodes inspected during a nearest neighbor query by Annoy. The higher,
        the more computation time required, but the higher the accuracy. The default is `-1`, which
        sets `search_k = n_trees * k`, where k is the number of neighbors to retrieve. If this is
        set too low, a variable number of neighbors may be retrieved per data-point (default -1)
    model: str or keras.models.Model
        The keras model to train using triplet loss. If a model object is provided, an embedding
        layer of size 'embedding_dims' will be appended to the end of the network. If a string, a
        pre-defined network by that name will be used. Possible options are:
            * 'szubert'
            * 'hinton'
            * 'maaten'
        (default 'szubert')
    supervision_metric: str or function
        The supervision metric to optimize when training keras in supervised mode. Supports all of
        the classification losses included with keras, so long as the labels are provided in the
        correct format. A list of keras' loss functions can be found at https://keras.io/losses/
        (default 'sparse_categorical_crossentropy')
    supervision_weight: float
        Float between 0 and 1 denoting the weighting to give to classification vs triplet loss when
        training in supervised mode. The higher the weight, the more classification influences
        training (default 0.25)
    annoy_index_path: str or Path
        The filepath of a pre-trained annoy index file to be saved on disk (default 'annoy.index')
    callbacks: list[keras.callbacks.Callback]
        List of keras Callbacks to pass model during training, such as the TensorBoard callback.
        Note these only apply to `sub_epochs` in each `epoch` before hard negative examples are
        re-computed (default [])
    early_stopping: bool
        Stop training the model if loss does not increase in `sub_epochs - 1` in any epoch in
        `epochs` (default True)
    reduce_lr_amount: float
        Amount to reduce learning rate by after each completed full epoch (default 1.)
    verbose: bool
        Controls the volume of logging output the model produces when training. When set to `False`,
        silences outputs, else will print outputs (default True)

    """
    def __init__(
        self,
        embedding_dims=2,
        k=150,
        distance='pn',
        batch_size=128,
        epochs=20,
        sub_epochs=10,
        margin=1.,
        ntrees=50,
        search_k=-1,
        model='szubert',
        supervision_metric='sparse_categorical_crossentropy',
        supervision_weight=0.25,
        annoy_index_path='annoy.index',
        callbacks=[],
        early_stopping=True,
        reduce_lr_amount=1.,
        verbose=True,
    ):
        self.embedding_dims = embedding_dims
        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.sub_epochs = sub_epochs
        self.margin = margin
        self.ntrees = ntrees
        self.search_k = search_k
        self.model_def = model
        self.supervision_metric = supervision_metric
        self.supervision_weight = supervision_weight
        self.annoy_index_path = annoy_index_path
        self.callbacks = callbacks
        self.early_stopping = early_stopping
        self.reduce_lr_amount = reduce_lr_amount
        self.verbose = int(verbose)

        self.model_ = None
        self.encoder = None
        self.supervised_model_ = None
        self.loss_history_ = list()

        self.build_index_on_disk = True if platform.system() != 'Windows' else False

    def __getstate__(self):
        """Return object serializable variable dict."""
        state = dict(self.__dict__)
        if 'model_' in state:
            state['model_'] = None
        if 'encoder' in state:
            state['encoder'] = None
        if 'supervised_model_' in state:
            state['supervised_model_'] = None
        if 'callbacks' in state:
            state['callbacks'] = []
        if not isinstance(state['model_def'], str):
            state['model_def'] = None
        if 'neighbor_matrix' in state:
            state['neighbor_matrix'] = None
        return state

    def _fit(self, X, Y=None, shuffle_mode=True):
        if self.verbose > 0:
            print('Building KNN index...')
        build_annoy_index(
            X,
            self.annoy_index_path,
            ntrees=self.ntrees,
            build_index_on_disk=self.build_index_on_disk,
            verbose=self.verbose,
        )

        # sets up the triplet data
        datagen = generator_from_index(
            X,
            Y,
            index_path=self.annoy_index_path,
            k=self.k,
            batch_size=self.batch_size,
            search_k=self.search_k,
            verbose=self.verbose,
        )

        # setting up siamese network component
        loss_monitor = 'loss'
        try:
            triplet_loss_func = triplet_loss(distance=self.distance, margin=self.margin)
        except KeyError:
            raise ValueError(
                'Loss function `{}` not implemented.'.format(self.distance)
            )

        if type(self.model_def) is str:
            input_size = (X.shape[-1],)
            self.model_, anchor_embedding, _, _ = triplet_network(
                base_network(self.model_def, input_size),
                embedding_dims=self.embedding_dims,
            )
        else:
            self.model_, anchor_embedding, _, _ = triplet_network(
                self.model_def, embedding_dims=self.embedding_dims
            )

        # setting up supervised model component
        if not is_multiclass(self.supervision_metric):
            if not is_hinge(self.supervision_metric):
                # Binary logistic classifier
                if len(Y.shape) > 1:
                    self.n_classes = Y.shape[-1]
                else:
                    self.n_classes = 1
                supervised_output = Dense(
                    self.n_classes, activation='sigmoid', name='supervised'
                )(anchor_embedding)
            else:
                # Binary Linear SVM output
                if len(Y.shape) > 1:
                    self.n_classes = Y.shape[-1]
                else:
                    self.n_classes = 1
                supervised_output = Dense(
                    self.n_classes,
                    activation='linear',
                    name='supervised',
                    kernel_regularizer=regularizers.l2(),
                )(anchor_embedding)
        else:
            if not is_hinge(self.supervision_metric):
                validate_sparse_labels(Y)
                self.n_classes = len(np.unique(Y[Y != np.array(-1)]))
                # Softmax classifier
                supervised_output = Dense(
                    self.n_classes, activation='softmax', name='supervised'
                )(anchor_embedding)
            else:
                self.n_classes = len(np.unique(Y, axis=0))
                # Multiclass Linear SVM output
                supervised_output = Dense(
                    self.n_classes,
                    activation='linear',
                    name='supervised',
                    kernel_regularizer=regularizers.l2(),
                )(anchor_embedding)

        supervised_loss = keras.losses.get(self.supervision_metric)
        if self.supervision_metric == 'sparse_categorical_crossentropy':
            supervised_loss = semi_supervised_loss(supervised_loss)

        final_network = Model(
            inputs=self.model_.inputs, outputs=[self.model_.output, supervised_output]
        )
        self.model_ = final_network
        self.model_.compile(
            optimizer='adam',
            loss={'stacked_triplets': triplet_loss_func, 'supervised': supervised_loss},
            loss_weights={
                'stacked_triplets': 1 - self.supervision_weight,
                'supervised': self.supervision_weight,
            },
        )

        # Store dedicated classification model
        supervised_model_input = Input(shape=(X.shape[-1],))
        embedding = self.model_.layers[3](supervised_model_input)
        softmax_out = self.model_.layers[-1](embedding)

        self.supervised_model_ = Model(supervised_model_input, softmax_out)

        self.encoder = self.model_.layers[3]

        if self.verbose > 0:
            print('Training neural network...')

        if self.early_stopping:
            early_stopping_monitor = EarlyStopping(
                monitor=loss_monitor, patience=self.sub_epochs - 1,
            )
            self.callbacks.append(early_stopping_monitor)

        for epoch in tqdm(range(0, self.epochs)):
            if epoch > 0:
                data_to_annoy = self.transform(X, verbose=False)

                if self.verbose:
                    print('Building new KNN index...')
                build_annoy_index(
                    data_to_annoy,
                    self.annoy_index_path,
                    ntrees=self.ntrees,
                    build_index_on_disk=self.build_index_on_disk,
                    verbose=0,
                )

                if self.verbose:
                    print('Extracting KNN from new index...')
                neighbor_matrix = extract_knn(
                    X=data_to_annoy,
                    index_filepath=self.annoy_index_path,
                    k=self.k,
                    search_k=self.search_k,
                    verbose=0
                )

                assert X.shape[0] == neighbor_matrix.shape[0]
                datagen.neighbor_matrix = neighbor_matrix
                datagen.hard_mode = (epoch / self.epochs)

                if self.verbose > 0:
                    print('Training next epoch...')

            hist = self.model_.fit_generator(
                datagen,
                epochs=self.sub_epochs,
                callbacks=self.callbacks,
                shuffle=shuffle_mode,
                steps_per_epoch=int(np.ceil(X.shape[0] / self.batch_size)),
                verbose=0,
            )

            if early_stopping_monitor.stopped_epoch > 0:
                print('Early stopping.')
                break

            if os.path.exists(self.annoy_index_path):
                os.remove(self.annoy_index_path)

            mean_loss = np.mean(hist.history['loss'][-self.sub_epochs:])
            print('Epoch {}: loss {}'.format(epoch, mean_loss))

            if self.reduce_lr_amount != 1:
                K.set_value(self.model_.optimizer.learning_rate,
                            K.eval(self.model_.optimizer.lr) * self.reduce_lr_amount)
                if self.verbose:
                    print('New learning rate set to: {}'.format(K.eval(self.model_.optimizer.lr)))

        self.loss_history_ += hist.history['loss']

    def fit(self, X, Y, shuffle_mode=True):
        """
        Fit an meezer model.

        Parameters
        ----------
        X: np.array, shape (n_samples, n_features)
            Data to be embedded.
        Y: np.array, shape (n_samples)
            Labels for data in `X`. If `Y` contains -1 labels, and 'sparse_categorical_crossentropy'
            is the loss function, semi-supervised learning will be used.
        shuffle_mode: bool
            Parameter to be sent to Keras `fit_generator` function (default True)

        """
        self._fit(X, Y, shuffle_mode)

        return self

    def fit_transform(self, X, Y=None, shuffle_mode=True):
        """
        Fit to data then transform

        Parameters
        ----------
        X: np.array, shape (n_samples, n_features)
            Data to be embedded.
        Y: np.array, shape (n_samples)
            Labels for data in `X`. If `Y` contains -1 labels, and 'sparse_categorical_crossentropy'
            is the loss function, semi-supervised learning will be used.
        shuffle_mode: bool
            Parameter to be sent to Keras `fit_generator` function (default True)

        Returns
        -------
        X_new : np.array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.

        """
        self.fit(X, Y, shuffle_mode)

        return self.transform(X)

    def transform(self, X, verbose=True):
        """
        Transform X into the existing embedded space and return that
        transformed output.

        Parameters
        ----------
        X: np.array, shape (n_samples, n_features)
            New data to be transformed.
        verbose: bool

        Returns
        -------
        X_new: np.array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.

        """
        embeddings = self.encoder.predict(X, verbose=verbose)

        return embeddings

    def score_samples(self, X):
        """
        Passes X through classification network to obtain predicted supervised values.

        Parameters
        ----------
        X: np.array, shape (n_samples, n_features)
            Data to be passed through classification network.

        Returns
        -------
        X_new: np.array, shape (n_samples, embedding_dims)
            Softmax class probabilities of the data.

        """
        softmax_output = self.supervised_model_.predict(X, verbose=self.verbose)

        return softmax_output

    def save_model(self, folder_path, overwrite=False):
        """
        Save an meezer model

        Parameters
        ----------
        folder_path: string
            Path to serialised model files and metadata
        overwrite: bool
            Whether or not to overwrite existing data (default False)

        Side Effects
        -------------
        Writes model files to disk.

        """
        if overwrite:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        # serialize weights to HDF5
        self.model_.save(os.path.join(folder_path, 'meezer_model.h5'))
        # Have to serialize supervised model separately
        if self.supervised_model_ is not None:
            self.supervised_model_.save(
                os.path.join(folder_path, 'supervised_model.h5')
            )

        json.dump(
            self.__getstate__(),
            open(os.path.join(folder_path, 'meezer_params.json'), 'w'),
        )

    def load_model(self, folder_path):
        """
        Load meezer model

        Parameters
        ----------
        folder_path: string
            Path to serialised model files and metadata

        Returns
        -------
        model: Meezer instance

        """
        meezer_config = json.load(
            open(os.path.join(folder_path, 'meezer_params.json'), 'r')
        )
        self.__dict__ = meezer_config

        loss_function = triplet_loss(self.distance, self.margin)
        self.model_ = load_model(
            os.path.join(folder_path, 'meezer_model.h5'),
            custom_objects={'tf': tf, loss_function.__name__: loss_function},
        )
        self.encoder = self.model_.layers[3]
        self.encoder._make_predict_function()

        # If a supervised model exists, load it
        supervised_path = os.path.join(folder_path, 'supervised_model.h5')
        if os.path.exists(supervised_path):
            self.supervised_model_ = load_model(supervised_path)
        return self
