"""
This module will contain the mixins providing algorithms and fields which are common to many filtering algorithms.
"""

import numpy as np
from numbers import Number


class GaussianMixtureMixin():
    """
    This mixin provides the behavior of algorithms working with Gaussian mixtures. It provides the fields, which enable
    the class to be considered a Gaussian mixture. Further an algorithms for merging and capping Gaussian mixture
    components are provided
    """

    def __init__(self):
        self._weights = None
        self._means = None
        self._covariances = None

    @property
    def weights(self):
        return self._weights

    @property
    def means(self):
        return self._means

    @property
    def covariances(self):
        return self._covariances

    def reduce(self, threshold, distance):
        """
        Reduces the number of mixture components by merging similar components

        This method reduces the number of components present in Gaussian mixture by merging those components whose
        mutual Mahalanobis distance is less then a given parameter. Only components whose weights are above threshold
        are considered for merging, while components whose weights are under threshold are discarded.

        This method implements the algorithm outlined in :

        B.-N. Vo, and W. K. Ma, "The Gaussian mixture Probability Hypothesis Density Filter"
        IEEE Trans Signal Processing, Vol. 54, No. 11, pp. 4091-4104, 2006

        :param threshold: number
        :param distance: number
        :return: void
        """

        if not isinstance(threshold, Number):
            raise TypeError("The threshold passed in has to be a number")
        if not isinstance(distance, Number):
            raise TypeError("The distance passed in has to be a number")

        new_weights = []
        new_means = []
        new_covariances = []

        threshold_indices = np.squeeze(self._weights > threshold)
        self._weights = self._weights[:, threshold_indices]
        self._means = self._means[threshold_indices, :]
        self._covariances = self._covariances[threshold_indices, :, :]

        indices = np.arange(self._weights.size)
        while indices.size > 0:
            argmax_index = indices[np.argmax(self._weights)]

            current_mean = self._means[argmax_index, :]
            current_cov = self._covariances[argmax_index, :, :]

            # compute the mahalanobis distance between current mean
            # and all other means
            diff = self._means - current_mean
            distances = np.sqrt(np.sum(diff.T * (current_cov @ diff.T), axis=0))
            take_mask = distances < distance
            selected_weights = self._weights[:, take_mask]
            selected_states = self._means[take_mask, :]
            selected_covariances = self._covariances[take_mask, :, :]

            new_weight = np.sum(selected_weights, axis=1)
            new_state = (1 / new_weight) * np.sum(selected_weights.T * selected_states, axis=0)

            diff = new_state - selected_states
            add_cov = selected_covariances + diff[:, :, np.newaxis] @ diff[:, np.newaxis, :]
            new_covariance = (1 / new_weight) * np.sum((selected_weights.T)[:, np.newaxis] * add_cov, axis=0)

            iter_mask = ~take_mask
            indices = np.arange(indices[~take_mask].size)
            self._weights = self._weights[:, iter_mask]
            self._means = self._means[iter_mask, :]
            self._covariances = self._covariances[iter_mask, :, :]

            new_weights.append(new_weight)
            new_means.append(new_state)
            new_covariances.append(new_covariance)

        self._weights = np.reshape(np.asarray(new_weights), (1, -1))
        self._means = np.asarray(new_means)
        self._covariances = np.asarray(new_covariances)


    def cap(self, number):
        """
        Limits the number of components present in Gaussian mixture.

        If number of components requested is larger then number of components present, then nothing happens. Otherwise
        the components are sorted in descending order by weights, and only so many components are kept as requested.

        :param number: number of components to keep
        :return:
        """

        if not isinstance(number, Number):
            raise TypeError("The number of components request has to be a number")

        if number > self._weights.size:
            return

        descending_sorting_indices = np.squeeze(np.argsort(-self._weights))
        descending_weights = self._weights[:, descending_sorting_indices]
        descending_means = self._means[descending_sorting_indices, :]
        descending_covariances = self._covariances[descending_sorting_indices, :, :]
        self._weights = descending_weights[:, 0:number]
        self._means = descending_means[0:number, :]
        self._covariances = descending_covariances[0:number, :, :]
