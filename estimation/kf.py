import numpy as np
from estimation.kalman import Kalman


class KF(Kalman):

    def __init__(self,
                 transition_model,
                 transition_noise,
                 measurement_model,
                 measurement_noise,
                 states=None,
                 covariances=None):

        self._transition_model = transition_model
        self._transition_noise = transition_noise
        self._measurement_model = measurement_model
        self._measurement_noise = measurement_noise

        self.states = states
        self.covariances = covariances

    def predict(self):
        """
        Computes the predict step of classical Kalman filter

        :return: None

        Usage: This function can be used to create compute prediction of many states at once. Limitation to
        vectorization is that all states are predicted with same transition model. It is also assumed that there is
        only single control vector which is applied to all predicted states.
        """

        self.states = self.states @ self._transition_model.T
        self.covariances = self._transition_model @ self.covariances @ self._transition_model.T

    def update(self, measurements):
        """
        :param measurements: Numpy array for storing measurements, where each row is treated as a measurement.

        :return: Returns updated states and covariance matrices
        """

        # matrix times vector operations are transposed because the states and measurements are viewed as stored in rows
        if self._measurement_noise.ndim < 3:
            self._measurement_noise = self._measurement_noise[np.newaxis]

        innovation = measurements - self.states @ self._measurement_model.T

        _, _, kalman_gain = self.compute_update_matrices()

        self.pure_update(innovation, kalman_gain)

    def compute_update_matrices(self):

        innovation_covariance = self._measurement_noise +\
                                self._measurement_model @ self.covariances @ self._measurement_model.T
        inv_innovation_covariances = np.linalg.inv(innovation_covariance)
        kalman_gain = self.covariances @ self._measurement_model.T @ inv_innovation_covariances
        return innovation_covariance, inv_innovation_covariances, kalman_gain

    def pure_update(self, innovation, kalman_gain):
        dim = self.states.shape[1]
        self.states = self.states + np.squeeze(innovation[:, np.newaxis] @ np.transpose(kalman_gain, (0, 2, 1)))
        self.covariances = (np.eye(dim) - kalman_gain @ self._measurement_model) @ self.covariances