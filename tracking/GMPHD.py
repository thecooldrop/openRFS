import numpy as np
from estimation.kf import Kalman


class GMPHD:

    def __init__(self,
                 weights,
                 states,
                 covariances,
                 birth_weights,
                 birth_states,
                 birth_covariances,
                 spawn_weights,
                 spawn_offsets,
                 spawn_covariances,
                 spawn_transition,
                 kalman_filter,
                 probability_detection,
                 probability_survival,
                 clutter_lambda,
                 clutter_density):
        """
         Computes complete process of GMPHD filter as described in paper by Ba-Ngu Vo
         Parameters:
         -----------
         weights: np.ndarray
                 Prior weights specified as a row vector
         states: np.ndarray
                 Prior states specified as two dimensional np.ndarray, where each row represents a prior state
         covariance: np.ndarray
                 Prior covariances specified as three dimensional np.ndarray, where each entry in first dimension is a
                 covariance matrix for matching state
         birth_weights: np.ndarray
                 Weights for birth components given as a row vector
         birth_states: np.ndarray
                 Birth states specified as two dimensional np.ndarray, where each row represents a birth state
         birth_covariances: np.ndarray
                 Covariances of birth states specified as three dimensional np.ndarray, where each entry in first
                 dimension is a covariance matrix for matching birth state
         spawn_weights: np.ndarray
                 Weights of spawn components specified as a row vector
         spawn_offsets: np.ndarray
                 A two dimensional np.ndarray of offsets relative to state, where spawn components are born. Each row is
                 considered an offset vector.
         spawn_covariances: np.ndarray
                 Covariance matrices for spawn components specified as three dimensional np.ndarray.
         spawn_transition: np.ndarray
                 A three dimensional np.ndarray of transition matrices. Each entry along first dimension is considered a
                 transition matrix.
         probability_detection: float
                 A numeric value between 0 and 1, used to specify a probability that a state is measured
         probability_survival: float
                 A numeric value between 0 and 1, used to specify that target will continue to exist between two
                 measurements
         clutter_lambda: float
                 A positive numeric value specifying the number of expected false alarm measurements in current
                 time-step
         clutter_density: float
                 Inverse of the volume of the subset of measurement space observed by sensor, since the false alarm
                 density is modeled as uniform distributed over this subset.
         """

        self._weights = weights
        self._birth_weights = birth_weights
        self._birth_states = birth_states
        self._birth_covariances = birth_covariances
        self._spawn_weights = spawn_weights
        self._spawn_offsets = spawn_offsets
        self._spawn_covariances = spawn_covariances
        self._spawn_transition = spawn_transition
        self._probability_detection = probability_detection
        self._probability_survival = probability_survival
        self._clutter_lambda = clutter_lambda
        self._clutter_density = clutter_density

        self._kalman_filter: Kalman = kalman_filter
        self._kalman_filter.states = states
        self._kalman_filter.covariances = covariances

    def process(self,
                measurements):
        new_spawn_weights, new_spawn_states, new_spawn_covs = self.compute_spawn_components()

        # here internal state of PHD filter changes
        prediction_weights, prediction_states, prediction_covariances = self.predict_surviving_components()

        prediction_weights = np.concatenate((prediction_weights, self._birth_weights, new_spawn_weights), axis=1)
        prediction_states = np.concatenate((prediction_states, self._birth_states, new_spawn_states), axis=0)
        prediction_covariances = np.concatenate((prediction_covariances, self._birth_covariances, new_spawn_covs),
                                                axis=0)

        miss_weights, miss_states, miss_covariances = self.compute_miss_components()

        updated_weights, updated_states, updated_covariances = self.update_components(prediction_weights,
                                                                                      prediction_states,
                                                                                      prediction_covariances,
                                                                                      measurements)

        full_weights = np.concatenate((updated_weights, miss_weights), axis=1)
        full_states = np.concatenate((updated_states, miss_states), axis=1)
        full_covariances = np.concatenate((updated_covariances, miss_covariances), axis=0)
        # Here the internal state of PHD filter changes
        self._weights = full_weights
        self._kalman_filter.state = full_states
        self._kalman_filter.covariances = full_covariances

    def compute_spawn_components(self):
        # multiply each old weight with each of spawn_weights
        new_weights = np.reshape(self._weights.T @ self._spawn_weights, (1, -1))

        # compute transition of each state with each of transition matrices
        transit_states = self._spawn_transition @ self._kalman_filter.states.T
        # for each transition matrix add the corresponding offset vector
        transit_states = self._spawn_offsets[:, :, np.newaxis] + transit_states
        # reorder the axes, so that the second axis becomes the fastest varying and first second fastest varying
        # goal of this transpose is to make n-th entry along first dimension correspond to transitions from n-th state
        # and to make the m-th index along second axis be a row of n-th state and m-th state
        transit_states = np.transpose(transit_states, (2, 0, 1))
        # reshape the transit_states so that each row represents a state
        new_states = np.reshape(transit_states, (new_weights.shape[1], -1))

        # compute covariance for each new state and weight
        transit_covs = self._spawn_covariances[:, np.newaxis] + \
                       self._spawn_transition @ self._kalman_filter.covariances[:, np.newaxis] @ np.transpose(
            self._spawn_transition,
            (0, 2, 1))
        # reshape the covaraince into old 3D shape
        cov_shape_x, cov_shape_y = self._kalman_filter.covariances.shape
        new_covs = np.reshape(transit_covs, (-1, cov_shape_x, cov_shape_y))

        return new_weights, new_states, new_covs

    def predict_surviving_components(self):
        new_weights = self._probability_survival * self._weights
        self._kalman_filter.predict()
        new_states = self._kalman_filter.states
        new_covariances = self._kalman_filter.covariances
        return new_weights, new_states, new_covariances

    def compute_miss_components(self):
        new_weights = (1 - self._probability_detection) * self._weights
        states = self._kalman_filter.states
        covariances = self._kalman_filter.covariances
        return new_weights, states, covariances

    def update_components(self,
                          weights,
                          states,
                          covariances,
                          measurements):

        self._kalman_filter.states = states
        self._kalman_filter.covariances = covariances
        self._kalman_filter.update(measurements)
        inno_cov = self._kalman_filter.innovation_covariances
        inv_inno_cov = self._kalman_filter.inv_innovation_covariances
        num_meas = measurements.shape[0]
        num_states = states.shape[0]
        mult_weights = np.repeat(weights, num_meas, 1)

        updated_states = self._kalman_filter.states
        updated_covariances = self._kalman_filter.inv_innovation_covariances
        # compute difference between each measurement and each state

        inno_dets = np.reshape(np.linalg.det(2 * np.pi * inno_cov), (1, -1))
        innovations = self._kalman_filter.innovation
        exponent = innovations[:, np.newaxis] @ inv_inno_cov @ np.transpose(innovations[:, np.newaxis],
                                                                                     (0, 2, 1))
        exponent = np.reshape(exponent, (1, -1))
        gaussians = (inno_dets ^ (-1 / 2)) * np.exp((-1 / 2) * exponent)

        scaled_weights = self._probability_detection * mult_weights * gaussians
        l_weights = np.reshape(scaled_weights, (num_meas, num_states))
        sum_weights = np.sum(l_weights, axis=1)
        sum_weights = np.reshape(sum_weights, (-1, 1))
        coeffs = self._clutter_lambda * self._clutter_density + sum_weights
        l_up_weights = l_weights / coeffs
        updated_weights = np.reshape(l_up_weights, (1, -1))

        return updated_weights, updated_states, updated_covariances
