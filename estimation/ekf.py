import numpy as np
from estimation.kalman import Kalman


class EKF(Kalman):

    def __init__(self,
                 transition_model,
                 transition_jacobi,
                 transition_noise,
                 measurement_model,
                 measurement_jacobi,
                 measurement_noise,
                 states=None,
                 covariances=None):

        self._transition_model = transition_model
        self._transition_noise = transition_noise
        self._transition_jacobi = transition_jacobi
        self._measurement_model = measurement_model
        self._measurement_noise = measurement_noise
        self._measurement_jacobi = measurement_jacobi

        self.states = states
        self.covariances = covariances

    def predict(self):
        """
        :return: None
        """

        # jacobis is 3D matrix, since each of them is linearized about different state
        jacobis = self._transition_jacobi(self.states)
        self.states = self._transition_model(self.states)
        self.covariances = jacobis @ self.covariances @ jacobis.transpose((0, 2, 1)) + self._transition_noise

    def update(self, measurements):
        """
        :param measurements:
        :return: None
        """
        # ensure states is in row form
        expected_measurements = self._measurement_model(self.states)
        innovation = measurements - expected_measurements
        jacobis, _, _, kalman_gain = self.compute_update_matrices()
        self.pure_update(innovation, kalman_gain)

    def compute_update_matrices(self):

        jacobis = self._measurement_jacobi(self.states)
        innovation_covariance = self._measurement_noise + jacobis @ self.covariances @ np.transpose(jacobis, (0, 2, 1))
        inv_innovation_covariance = np.linalg.inv(innovation_covariance)
        kalman_gain = self.covariances @ np.transpose(jacobis, (0, 2, 1)) @ inv_innovation_covariance
        return jacobis, innovation_covariance, inv_innovation_covariance, kalman_gain

    def pure_update(self,
                    innovation,
                    kalman_gain,
                    innovation_covariances=None):

        dim = self.states.shape[1]
        self.states = self.states + np.squeeze(innovation[:, np.newaxis] @ np.transpose(kalman_gain, (0, 2, 1)))
        jacobis = self._measurement_jacobi(self.states)
        self.covariances = (np.eye(dim) - kalman_gain @ jacobis) @ self.covariances
