import numpy as np
from estimation.kalman import Kalman


class UKF(Kalman):

    def __init__(self,
                 transition_model,
                 transition_jacobi,
                 transition_noise,
                 measurement_model,
                 measurement_jacobi,
                 measurement_noise,
                 alpha,
                 beta,
                 ket,
                 states=None,
                 covariances=None,
                 innovation=None,
                 innovation_covariances=None,
                 kalman_gains=None,
                 ):
        self._transition_model = transition_model
        self._transition_jacobi = transition_jacobi
        self._transition_noise = transition_noise
        self._measurement_model = measurement_model
        self._measurement_jacobi = measurement_jacobi
        self._measurement_noise = measurement_noise
        self._alpha = alpha
        self._beta = beta
        self._ket = ket
        self._innovation = innovation
        self._innovation_covariances = innovation_covariances
        self._kalman_gains = kalman_gains
        self._states = states
        self._covariances = covariances

        self._mean_measurements = None

    def predict(self):
        dim = self._states.shape[1]
        lmbd = self._alpha ** 2 * (dim + self._ket) - dim

        sigma_points = self._compute_sigma_points(lmbd, dim)
        c_weights, m_weights = self._compute_sigma_weights(lmbd, dim)

        predictions = self._transition_model(sigma_points)
        self._states = np.sum(m_weights[:, np.newaxis] * predictions, axis=1)

        deviation = predictions - self._states[:, np.newaxis]
        scaled_deviation = c_weights[:, np.newaxis] * deviation
        self._covariances = np.sum(scaled_deviation[:, :, :, np.newaxis] @ deviation[:, :, np.newaxis],
                                   axis=1) + self._transition_noise

    def update(self, measurements):
        self.compute_update_matrices()
        self._innovation = measurements - self._mean_measurements
        self.pure_update()

    def compute_update_matrices(self):
        dim = self._states.shape[1]
        lmbd = self._alpha ** 2 * (dim + self._ket) - dim

        sigma_points = self._compute_sigma_points(lmbd, dim)
        c_weights, m_weights = self._compute_sigma_weights(lmbd, dim)

        expected_measurements = self._measurement_model(sigma_points)
        self._mean_measurements = np.sum(m_weights[:, np.newaxis] * sigma_points, axis=1)

        deviation = expected_measurements - self._mean_measurements[:, np.newaxis]
        scaled_deviation = c_weights[:, np.newaxis] * deviation
        self._innovation_covariances = np.sum(scaled_deviation[:, :, :, np.newaxis] @ deviation[:, :, np.newaxis],
                                              axis=1) + self._measurement_noise

        sigma_deviation = sigma_points - self._states[:, np.newaxis]
        scaled_sigma_deviation = c_weights[:, np.newaxis] * sigma_deviation
        kalman_factor = np.sum(scaled_sigma_deviation[:, :, :, np.newaxis] @ sigma_deviation[:, :, np.newaxis], axis=1)

        inv_inno = np.linalg.inv(self._innovation_covariances)
        self._kalman_gains = kalman_factor @ inv_inno

    def pure_update(self):
        self._states = self._states + np.squeeze(
            self._innovation[:, np.newaxis] @ np.transpose(self._kalman_gains, (0, 2, 1)))
        self._covariances = self._covariances - \
                            self._kalman_gains @ self._innovation_covariances @ np.transpose(self._kalman_gains,
                                                                                             (0, 2, 1))

    def _compute_sigma_weights(self, lmbd, n):
        weight: float = lmbd / (n + lmbd)
        m_weights = np.repeat(weight, 2 * n + 1)
        c_weights = np.repeat(weight, 2 * n + 1)
        c_weights[0] = c_weights[0] + (1 - self._alpha ** 2 + self._beta)
        c_weights[1:] = (1 / (2 * lmbd)) * c_weights[1:]
        m_weights[1:] = (1 / (2 * lmbd)) * m_weights[1:]
        return c_weights, m_weights

    def _compute_sigma_points(self, lmbd, n):
        root_covariances = np.linalg.cholesky((lmbd + n) * self._covariances)
        # To each state we compute a matrix of sigma points
        sig_first = self._states[:, np.newaxis] + np.transpose(root_covariances, (0, 2, 1))
        sig_second = self._states[:, np.newaxis] - np.transpose(root_covariances, (0, 2, 1))
        sigma_points = np.concatenate((self._states[:, np.newaxis], sig_first, sig_second), axis=1)
        return sigma_points
