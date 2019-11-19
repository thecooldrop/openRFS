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
                 covariances=None,
                 innovation=None,
                 innovation_covariances=None,
                 inv_innovation_covariances=None,
                 kalman_gains=None):
        super(EKF, self).__init__(transition_model,
                                  transition_noise,
                                  measurement_model,
                                  measurement_noise,
                                  states,
                                  covariances,
                                  innovation,
                                  innovation_covariances,
                                  inv_innovation_covariances,
                                  kalman_gains)
        self._transition_jacobi = transition_jacobi
        self._measurement_jacobi = measurement_jacobi

    def predict(self):
        """
        :return: None
        """

        # jacobis is 3D matrix, since each of them is linearized about different state
        jacobis = self._transition_jacobi(self._states)
        self._states = self._transition_model(self._states)
        self._covariances = jacobis @ self._covariances @ jacobis.transpose((0, 2, 1)) + self._transition_noise

    def update(self, measurements):
        """
        :param measurements:
        :return: None
        """
        # ensure states is in row form
        num_meas = measurements.shape[0]
        expected_measurements = self._measurement_model(self._states)
        self._innovation = measurements[:, np.newaxis, :] - expected_measurements
        self._innovation = np.transpose(self._innovation, (1, 0, 2))
        self.compute_update_matrices()
        self._covariances = np.repeat(self._innovation_covariances, num_meas, axis=0)
        self.pure_update()

    def compute_update_matrices(self):
        dim = self._states.shape[1]
        jacobis = self._measurement_jacobi(self._states)
        self._innovation_covariances = self._measurement_noise + \
                                       jacobis @ self._covariances @ np.transpose(jacobis, (0, 2, 1))

        self._inv_innovation_covariances = np.linalg.inv(self._innovation_covariances)
        self._kalman_gains = self._covariances @ np.transpose(jacobis, (0, 2, 1)) @ self._inv_innovation_covariances
        self._covariances = (np.eye(dim) - self._kalman_gains @ jacobis) @ self._covariances

    def pure_update(self):
        self._states = self._states[:, np.newaxis, :] + self._innovation @ np.transpose(self._kalman_gains, (0, 2, 1))
        self._states = np.concatenate(self._states[:])
