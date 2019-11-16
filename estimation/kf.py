import numpy as np
from estimation.kalman import Kalman


class KF(Kalman):

    def __init__(self,
                 transition_model,
                 transition_noise,
                 measurement_model,
                 measurement_noise,
                 states=None,
                 covariances=None,
                 innovation=None,
                 innovation_covariances=None,
                 inv_innovation_covariances=None,
                 kalman_gains=None):
        self._transition_model = transition_model
        self._transition_noise = transition_noise
        self._measurement_model = measurement_model
        self._measurement_noise = measurement_noise
        self._innovation = innovation
        self._innovation_covariances = innovation_covariances
        self._inv_innovation_covariances = inv_innovation_covariances
        self._kalman_gains = kalman_gains
        self._states = states
        self._covariances = covariances

    def predict(self):
        """
        Computes the predict step of classical Kalman filter

        :return: None

        Usage: This function can be used to compute prediction of many states at once. Limitation to
        vectorization is that all states are predicted with same transition model. It is also assumed that there is
        only single control vector which is applied to all predicted states.
        """

        self._states = self._states @ self._transition_model.T
        self._covariances = self._transition_model @ self._covariances @ self._transition_model.T

    def update(self, measurements):
        """
        :param measurements: Numpy array for storing measurements, where each row is treated as a measurement.

        :return: Returns updated states and covariance matrices
        """
        num_meas = measurements.shape[0]
        self._innovation = measurements[:, np.newaxis, :] - self._states @ self._measurement_model.T
        self._innovation = np.transpose(self._innovation, (1, 0, 2))
        self.compute_update_matrices()
        self._covariances = np.repeat(self._covariances, num_meas, axis=0)
        self.pure_update()

    def compute_update_matrices(self):
        dim = self._states.shape[1]
        self._innovation_covariances = self._measurement_noise + \
                                       self._measurement_model @ self._covariances @ self._measurement_model.T
        self._inv_innovation_covariances = np.linalg.inv(self._innovation_covariances)
        self._kalman_gains = self._covariances @ self._measurement_model.T @ self._inv_innovation_covariances
        self._covariances = (np.eye(dim) - self._kalman_gains @ self._measurement_model) @ self._covariances

    def pure_update(self):
        self._states = self._states[:, np.newaxis, :] + self._innovation @ np.transpose(self._kalman_gains, (0, 2, 1))
        self._states = np.concatenate(self._states[:])
