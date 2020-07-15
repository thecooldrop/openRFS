"""
This module contains the implementation of Gaussian Mixture Probability Hypothesis Density Filter as outlined described
in the paper by Ba-Ngu-Vo:

B.-N. Vo, and W. K. Ma, "The Gaussian mixture Probability Hypothesis Density Filter,"
IEEE Trans Signal Processing, Vol. 54, No. 11, pp. 4091-4104, 2006
"""

import numpy as np

from estimation.kalman import Kalman
from util.mixins import GaussianMixtureMixin
from numbers import Number


class GMPHD(GaussianMixtureMixin):
    """
    This class represents an instance of GMPHD filter. The class is meant to be instantiated with the Builder class
    provided as inner class. The formal description of the algorithm implemented here in pseudocode can be found in
    paper cited in the module documentation.

    This class is meant to be used iteratively by calling the process() method of the instance. The results of
    computations can be obtained after each call by calling the get_components() method of the class. As an example
    assume that measurements is already initialized to some valid input:
    >>> measurements = ...
    >>> gmphd_filter_instance = ...
    >>> gmphd_filter_instace.process(measurements)
    >>> weights, means, covariances = gmphd_filter_instance.get_components()

    """

    class Builder:
        """
        This is an inner builder class for instance of GMPHD filter. Purpose of this class is to make it easier to
        create instances of GMPHD filter and to document the relationships between variables in code and those in
        paper.

        The class should first be instantiated before the first use. After an instance of the class is created
        we can call the method to set the parameters of GMPHD filter, and after all paramters have been set an instance
        of GMPHD filter can be obtained by using the build() method.

        Each of the parameter-setting methods returns the instance of the class back, so that the class can either be
        used like regular class, or like a class with fluent interface. The classical usage would be:

        >>> gmphd_builder = GMPHD.Builder()
        >>> gmphd_builder = gmphd_builder.with_birth_covariances(birth_covariances)
        >>> gmphd_builder = gmphd_builder.with_birth_means(birth_means)
        >>> ...
        >>> gmphd_filter = gmphd_builder.build()

        On the other hand fluent interface would be something like:

        >>> gmphd_filter = GMPHD.Builder().with_birth_weights(birth_weights) \
                                          .with_birth_means(birth_states) \
                                          .with_birth_covariances(birth_covariances) \
                                          .with_probability_of_detection(p_det) \
                                          .with_probability_of_survival(p_surv) \
                                          .with_clutter_density(clutter_num) \
                                          .with_kalman_filter(kalman_filter) \
                                          .build()

        Note that the parameter-setting methods only check the passed in paremeters one by one, as the methods are
        called. The final check of compatibility of parameter values is done in build() method, which then either
        returns the built GMPHD filter or raises an error describing the parameters whose values are incompatible.

        """
        def __init__(self):
            self._weights = None
            self._means = None
            self._covariances = None
            self._birth_weights = None
            self._birth_means = None
            self._birth_covariances = None
            self._probability_detection = None
            self._probability_survival = None
            self._clutter_density = None
            self._kalman_filter = None

        def with_weights(self, weights):
            """
            Initializes the weights of Gaussian mixture density. The weights are numeric values representing the
            number of targets present for matching mean and covariance. The paper cited in module documentation
            denotes the weights with lowercase "w" with subscripts and superscripts.

            Note that the number of weights components should be equal to number of means and covariances given.
            Lastly it is expected that weights are given as a numpy array of numeric values
            :param weights: (1,n) numpy array of numeric
            :return: the builder instance
            """
            self._weights = weights
            return self

        def with_means(self, means):
            """
            Initializes the means of Gaussian components for the Gaussian mixture model. The means represent the actual
            mean values of Gaussian mixture components. The paper cited in module documentation denotes these means
            by lowercase "m" with subscripts and superscripts, and uses these variables in summations representing the
            Gaussian mixtures. For examples of usage of this variable in the algorithm refer to equations (23) and (31)

            Note that the number of mean components needs to be equal to both the number of weights and covariance
            components.

            Lastly it is expected that means be given as an numpy array, where each row represents a single mean vector.
            :param means: (m,n)-shaped numpy array, where n is the dimension of the mean
            :return: the current builder
            """
            self._means = means
            return self

        def with_covariances(self, covariances):
            """
            Initializes the covariances of the Gaussian components for the Gaussian mixture model. The covariances are
            the covariances of Gaussian mixture components. The paper cited in module documentation denotes these
            covariances with uppercase letter "P" with subscripts and superscripts, and uses these variables in
            summations representing the Gaussian mixtures. For examples of usage of this variable in the algorithm refer
            to equations (23) and (31) in the cited paper.

            Note that the number of covariance components needs to be equal to both the number of weights and covariance
            components.

            Lastly it is expected that the covariances be gives as a 3D numpy array, where entries along first dimension
            represent the covariance matrices
            :param covariances: (m, n, n)-shaped numpy array where the covariances are (n, n) shaped entries along first
                   axis
            :return: the current builder
            """
            self._covariances = covariances
            return self

        def with_birth_weights(self, birth_weights):
            """
            Initializes the weights of birth Gaussian mixture. The birth Gaussian mixture represents the expected
            born targets. The weights of this mixture represent the number of targets expected to be born with
            associated mean and covariance.

            In the paper the birth density is given lowercase greek gamma as can be seen in the equation (21) in the
            paper cited in the module documentation.

            The number of weights given should be equal to the number of birth means and birth covariances. Note that
            the birth weights should be given by numeric values. It is expected that these numeric values be given as
            numpy array.

            :param birth_weights: (1, n)-shaped numpy array of floating point numbers
            :return: the current builder
            """
            if birth_weights is None:
                raise ValueError("The birth weights of the GMPHD filter can not be None")
            else:
                self._birth_weights = birth_weights
            return self

        def with_birth_means(self, birth_means):
            """
            Initializes the means of the birth Gaussian mixture. These variables are denoted by lowercase "m" in the
            Gaussian birth mixture in the equation (21) in the cited paper.

            It is expected that birth means be arranged in an numpy array. Each row in the input array is expected to
            represent a single mean.

            :param birth_means: (m, n)-shaped numpy array
            :return: the current builder
            """
            if birth_means is None:
                raise ValueError("The birth states of the GMPHD filter can not be None")
            else:
                self._birth_means = birth_means
            return self

        def with_birth_covariances(self, birth_covariances):
            """
            Initializes the covariances of the birth Gaussian mixture. These variables are denoted by uppercase "P" in
            the Gaussian birth mixture in the equation (21) in the cited paper.

            It is expected that the birth covariance be arranged in a 3D numpy array. Each entry along the first axis
            is expected represent a single 2D covariance matrix.

            :param birth_covariances: (m, n, n)-shaped numpy array
            :return: the current builder instance
            """
            if birth_covariances is None:
                raise ValueError("The birth covariances of the GMPHD filter can not be None")
            else:
                self._birth_covariances = birth_covariances
            return self

        def with_probability_of_survival(self, probability_survival):
            """
            Initializes the probability that a single target remains present in the field of view between two
            measurements. This probability of survival is denoted by "p_s,k" ( lowercase p with "s,k" in the subscript )
            in the paper cited in module documentation. This variable is defined in equation (19) in the cited paper.

            This variable can take on values in interval [0,1]. The value 0 would mean that every observed target
            is not observable in the successive measurement because it escaped the field of view. On the other hand
            the value 1 would mean that every observed target is certain to remain within the field of view of the
            sensor in successive time interval.

            :param probability_survival: a numeric value in [0, 1] interval
            :return: the current builder
            """
            if not isinstance(probability_survival, Number):
                raise TypeError("The probability of detection for GMPHD filter has to be a number")
            if probability_survival is None:
                raise ValueError("The probability of detection for GMPHD filter can not be None")
            if probability_survival < 0 or probability_survival > 1:
                raise ValueError("The probability of survival for GMPHD filter has to be a value in [0, 1] interval")
            else:
                self._probability_survival = probability_survival
            return self

        def with_probability_of_detection(self, probability_detection):
            """
            Initializes the probability of detection for the filter. This variable denotes the probability that a target
            will cause a detection in the sensor, given that the target is present in the sensors field of view. In the
            cited paper this variable is denoted by "p_D,k" ( lowercase "p" with "D,k" in subscript ), and is defined
            in the equation (20) in the cited paper.

            This variable can take on values in interval [0, 1], where 0 would mean that targets present in the sensors
            field of view never cause any detections, while 1 would mean that every target present in the sensors field
            of view always causes a detection in the sensor.
            :param probability_detection: a numeric value in the [0, 1] interval
            :return: the current builder instance
            """
            if not isinstance(probability_detection, Number):
                raise TypeError("The probability of detection for GMPHD filter has to be a number")
            if probability_detection is None:
                raise ValueError("The probability of detection for GMPHD filter can not be None")
            if probability_detection < 0 or probability_detection > 1:
                raise ValueError("The probability of detection for GMPHD filter has to be a value in [0, 1] interval")
            else:
                self._probability_detection = probability_detection
            return self

        def with_clutter_density(self, clutter_density):
            """
            Initializes the clutter density for the GMPHD filter. This variable represents the average expected number
            of false positive measurements in each set of detections. Assume that that our sensor is a camera with
            pre-processing algorithm extracting the bounding boxes around pedestrians. If this sensor, on average,
            returns two extra bounding boxes which do not represent pedestrians then the clutter density would be 2.

            The value of this parameter should be set to ratio of expected number of false positive measurements to the
            volume of the measurement space. To illustrate how to compute this value assume we are still working with
            the camera sensor from above. Assume that that detections of camera consist of 4D vectors containing the
            [x, y, width, height] of bounding box. Further assume that for each detection 0 <= x <= 1920, 0<= y <= 1080
            , 0 <= width <= 400 and height 0 <= height <= 400. Under these assumptions the proper clutter density
            would be equal to (expected number of false positives ) / (1920*1080*400*400), where the divisor is
            the volume of observation space.

            In the cited paper this variable is denoted by lowercase lambda with "c" in subscript as can be seen in the
            equation (47).

            :param clutter_density: a positive numeric value
            :return: the current builder
            """
            if not isinstance(clutter_density, Number):
                raise TypeError("The clutter density for GMPHD filter has to be a number")
            if clutter_density is None:
                raise ValueError("The clutter density for GMPHD filter can not be None")
            if clutter_density < 0:
                raise ValueError("The clutter density for GMPHD filter has to be non-negative")
            else:
                self._clutter_density = clutter_density
            return self

        def with_kalman_filter(self, kalman):
            """
            Instantiates a Kalman filter for GMPHD filter.
            :param kalman: An object of type Kalman is expected
            :return: the current builder
            """
            if not isinstance(kalman, Kalman):
                raise TypeError("The Kalman Filter for GMPHD filter has to be an instance of estimation.Kalman")
            if kalman is None:
                raise ValueError("The Kalman filter for GMPHD filter can not be None")
            else:
                self._kalman_filter = kalman
            return self

        def build(self):
            """
            Builds an instance of GMPHD filter from the set values.

            Besides building an instance of GMPHD filter this method also checks that the values set for the builder
            are compatible. Following checks are done:

            - birth_weights, birth_means and birth_covariances are not None
            - number of birth weighs, means and covariances are all equal
            - probability of detection is not None
            - probability of survival is not None
            - clutter density is not None
            - Kalman filter is not None

            :return: an instace of GMPHD filter
            """
            if self._birth_weights is None:
                raise ValueError("The birth weights for GMPHD filter can not be None")
            if self._birth_means is None:
                raise ValueError("The birth means for GMPHD filter can not be None")
            if self._birth_covariances is None:
                raise ValueError("The birth covariances for GMPHD filter can not be None")

            birth_weights_number = self._birth_weights.shape[1]
            birth_means_number = self._birth_means.shape[0]
            birth_covariances_number = self._birth_covariances.shape[0]
            if not (birth_weights_number == birth_means_number and birth_means_number == birth_covariances_number):
                raise ValueError("The number of birth weights, birth means and birth covariances have to all be equal")

            # if any of initial parameters are None, then initialize them all to empty arrays of matching size
            if self._weights is None or self._means is None or self._covariances is None:
                self._weights = np.zeros((0, 1))
                self._means = np.zeros((0, self._birth_means.shape[1]))
                self._covariances = np.zeros((0, self._birth_covariances.shape[1], self._birth_covariances.shape[2]))

            if self._probability_detection is None:
                raise ValueError("The probability of detection for GMPHD filter can not be None")
            if self._probability_survival is None:
                raise ValueError("The probability of survival for GMPHD filter can not be None")
            if self._clutter_density is None:
                raise ValueError("The clutter density of GMPHD filter can not be None")

            if self._kalman_filter is None:
                raise ValueError("The Kalman filter for GMPHD filter can not be None")

            return GMPHD(self._weights,
                         self._means,
                         self._covariances,
                         self._birth_weights,
                         self._birth_means,
                         self._birth_covariances,
                         self._probability_detection,
                         self._probability_survival,
                         self._clutter_density,
                         self._kalman_filter)

    def __init__(self,
                 weights,
                 means,
                 covariances,
                 birth_weights,
                 birth_means,
                 birth_covariances,
                 probability_detection,
                 probability_survival,
                 clutter_density,
                 kalman_filter):

        super(GMPHD, self).__init__()
        self._weights = weights
        self._means = means
        self._covariances = covariances
        self._birth_weights = birth_weights
        self._birth_means = birth_means
        self._birth_covariances = birth_covariances
        self._probability_detection = probability_detection
        self._probability_survival = probability_survival
        self._clutter_density = clutter_density
        self._kalman_filter = kalman_filter

    def process(self, measurements):
        if self._weights.size > 0:
            self._kalman_filter.states = self._means
            self._kalman_filter.covariances = self._covariances
        else:
            self._kalman_filter.states = np.zeros((0, self._birth_means.shape[1]))
            self._kalman_filter.covariances = np.zeros((0, self._birth_covariances.shape[1], self._birth_covariances.shape[2]))
        predicted_weights, predicted_means, predicted_covariances = self._predict_existing_components()

        if predicted_weights.size > 0:
            predicted_weights = np.concatenate([self._birth_weights, predicted_weights], axis=1)
            predicted_means = np.concatenate([self._birth_means, predicted_means], axis=0)
            predicted_covariances = np.concatenate([self._birth_covariances, predicted_covariances], axis=0)
        else:
            predicted_weights = self._birth_weights
            predicted_means = self._birth_means
            predicted_covariances = self._birth_covariances

        self._kalman_filter.states = predicted_means
        self._kalman_filter.covariances = predicted_covariances
        update_means, update_covariances = self._update_components(measurements)

        miss_weights, miss_means, miss_covariances = self._miss_components(predicted_weights,
                                                                           predicted_means,
                                                                           predicted_covariances)

        q_matrix = self._compute_q_matrix()
        new_weights = self._update_weights(q_matrix, predicted_weights)

        self._weights = np.concatenate([miss_weights, new_weights], axis=1)
        self._means = np.concatenate([miss_means, update_means], axis=0)
        self._covariances = np.concatenate([miss_covariances, update_covariances], axis=0)

    def _predict_existing_components(self):
        weights = self._probability_survival * self._weights
        self._kalman_filter.predict()
        return weights, self._kalman_filter.states, self._kalman_filter.covariances

    def _update_components(self, measurements):
        self._kalman_filter.update(measurements)
        return self._kalman_filter.states, self._kalman_filter.covariances

    def _miss_components(self, weights, means, covariances):
        weights = (1 - self._probability_detection) * weights
        return weights, means, covariances

    def _compute_q_matrix(self):
        """
        The returned matrix is indexed by measurement and state, in that order, with scalars as values
        :return:
        """
        # innovations are indexed by state, then by measurement, so we need to transpose
        innovations = np.transpose(self._kalman_filter.innovations, (1, 0, 2))
        # number of inverse innovation covariances is equal to number of states
        inv_innovation_covariances = self._kalman_filter.inv_innovation_covariances
        innovation_covariances = self._kalman_filter.innovation_covariances

        first_factor = -(1 / 2)
        second_factor = innovations[:, :, np.newaxis, :]
        third_factor = inv_innovation_covariances
        fourth_factor = innovations[:, :, :, np.newaxis]
        first_product = np.exp(first_factor * second_factor @ third_factor @ fourth_factor)

        scaling_factors_inv = np.reshape(np.sqrt(np.linalg.det(2 * np.pi * innovation_covariances)), (1, -1))

        return (1 / scaling_factors_inv) * np.squeeze(first_product)

    def _update_weights(self, q_matrix, predicted_weights):
        first_factor = self._probability_detection * predicted_weights * q_matrix
        first_summand = self._clutter_density
        second_summand = np.sum(first_factor, axis=1)
        denominator = np.reshape(first_summand + second_summand, (-1, 1))
        new_weights = np.transpose(first_factor / denominator)
        return np.reshape(new_weights, (1, -1))

    def get_components(self):
        return np.copy(self._weights), np.copy(self._means), np.copy(self._covariances)