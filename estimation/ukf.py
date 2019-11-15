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
                 covariances=None):
        self._transition_model = transition_model
        self._transition_jacobi = transition_jacobi
        self._transition_noise = transition_noise
        self._measurement_model = measurement_model
        self._measurement_jacobi = measurement_jacobi
        self._measurement_noise = measurement_noise
        self._alpha = alpha
        self._beta = beta
        self._ket = ket

        self.states = states
        self.covariances = covariances

    def predict(self):
        dim = self.states.shape[1]
        lmbd = self._alpha ** 2 * (dim + self._ket) - dim

        sigma_points = self._compute_sigma_points(lmbd, dim)
        c_weights, m_weights = self._compute_sigma_weights(lmbd, dim)

        predictions = self._transition_model(sigma_points)
        self.states = np.sum(m_weights[:, np.newaxis] * predictions, axis=1)

        deviation = predictions - self.states[:, np.newaxis]
        scaled_deviation = c_weights[:, np.newaxis] * deviation
        self.covariances = np.sum(scaled_deviation[:, :, :, np.newaxis] @ deviation[:, :, np.newaxis],
                                  axis=1) + self._transition_noise

    def update(self, measurements):
        mean_measurement, innovation_covariances, _, kalman_gain = self.compute_update_matrices()
        innovation = measurements - mean_measurement
        self.pure_update(innovation, innovation_covariances, kalman_gain)

    def compute_update_matrices(self):
        dim = self.states.shape[1]
        lmbd = self._alpha ** 2 * (dim + self._ket) - dim

        sigma_points = self._compute_sigma_points(lmbd, dim)
        c_weights, m_weights = self._compute_sigma_weights(lmbd, dim)

        expected_measurements = self._measurement_model(sigma_points)
        mean_measurement = np.sum(m_weights[:, np.newaxis] * sigma_points, axis=1)

        deviation = expected_measurements - mean_measurement[:, np.newaxis]
        scaled_deviation = c_weights[:, np.newaxis] * deviation
        innovation_covariances = np.sum(scaled_deviation[:, :, :, np.newaxis] @ deviation[:, :, np.newaxis],
                                        axis=1) + self._measurement_noise

        sigma_deviation = sigma_points - self.states[:, np.newaxis]
        scaled_sigma_deviation = c_weights[:, np.newaxis] * sigma_deviation
        kalman_factor = np.sum(scaled_sigma_deviation[:, :, :, np.newaxis] @ sigma_deviation[:, :, np.newaxis], axis=1)

        inv_inno = np.linalg.inv(innovation_covariances)
        kalman_gain = kalman_factor @ inv_inno
        return mean_measurement, innovation_covariances, inv_inno, kalman_gain

    def pure_update(self,
                    innovation,
                    kalman_gain,
                    innovation_covariances=None):
        self.states = self.states + np.squeeze(innovation[:, np.newaxis] @ np.transpose(kalman_gain, (0, 2, 1)))
        self.covariances = self.covariances - \
                              kalman_gain @ innovation_covariances @ np.transpose(kalman_gain, (0, 2, 1))

    def _compute_sigma_weights(self, lmbd, n):
        weight: float = lmbd / (n + lmbd)
        m_weights = np.repeat(weight, 2 * n + 1)
        c_weights = np.repeat(weight, 2 * n + 1)
        c_weights[0] = c_weights[0] + (1 - self._alpha ** 2 + self._beta)
        c_weights[1:] = (1 / (2 * lmbd)) * c_weights[1:]
        m_weights[1:] = (1 / (2 * lmbd)) * m_weights[1:]
        return c_weights, m_weights

    def _compute_sigma_points(self, lmbd, n):
        root_covariances = np.linalg.cholesky((lmbd + n) * self.covariances)
        # To each state we compute a matrix of sigma points
        sig_first = self.states[:, np.newaxis] + np.transpose(root_covariances, (0, 2, 1))
        sig_second = self.states[:, np.newaxis] - np.transpose(root_covariances, (0, 2, 1))
        sigma_points = np.concatenate((self.states[:, np.newaxis], sig_first, sig_second), axis=1)
        return sigma_points
