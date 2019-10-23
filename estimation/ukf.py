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
                 ket):
        self.transition_model = transition_model
        self.transition_jacobi = transition_jacobi
        self.transition_noise = transition_noise
        self.measurement_model = measurement_model
        self.measurement_jacobi = measurement_jacobi
        self.measurement_noise = measurement_noise
        self.alpha = alpha
        self.beta = beta
        self.ket = ket

    def predict(self,
                states: np.ndarray,
                covariances: np.ndarray):
        """

        :param states:
        :param covariances:
        :return:

        Usage:
        Note that alpha can also be an array, so that each
        """

        dim = states.shape[1]
        lmbd = self.alpha ** 2 * (dim + self.ket) - dim

        sigma_points = self.compute_sigma_points(states, covariances, lmbd, dim)
        c_weights, m_weights = self.compute_sigma_weights(lmbd, dim)

        predictions = self.transition_model(sigma_points)
        predicted_state = np.sum(m_weights[:, np.newaxis] * predictions, axis=1)
        deviation = predictions - predicted_state[:, np.newaxis]
        scaled_deviation = c_weights[:, np.newaxis] * deviation
        predicted_covariance = np.sum(scaled_deviation[:, :, :, np.newaxis] @ deviation[:, :, np.newaxis],
                                      axis=1) + self.transition_noise
        return predicted_state, predicted_covariance

    def update(self,
               states,
               covariances,
               measurements):
        mean_measurement, innovation_covariances, _, kalman_gain = self.compute_update_matrices(states,
                                                                                                covariances)
        innovation = measurements - mean_measurement

        return self.pure_update(states, covariances, innovation, innovation_covariances, kalman_gain)

    def compute_sigma_weights(self, lmbd, n):
        weight: float = lmbd / (n + lmbd)
        m_weights = np.repeat(weight, 2 * n + 1)
        c_weights = np.repeat(weight, 2 * n + 1)
        c_weights[0] = c_weights[0] + (1 - self.alpha ** 2 + self.beta)
        c_weights[1:] = (1 / (2 * lmbd)) * c_weights[1:]
        m_weights[1:] = (1 / (2 * lmbd)) * m_weights[1:]
        return c_weights, m_weights

    def compute_sigma_points(self, states, covariances, lmbd, n):
        root_covariances = np.linalg.cholesky((lmbd + n) * covariances)
        # To each state we compute a matrix of sigma points
        sig_first = states[:, np.newaxis] + np.transpose(root_covariances, (0, 2, 1))
        sig_second = states[:, np.newaxis] - np.transpose(root_covariances, (0, 2, 1))
        sigma_points = np.concatenate((states[:, np.newaxis], sig_first, sig_second), axis=1)
        return sigma_points

    def compute_update_matrices(self,
                                states,
                                covariances):
        dim = states.shape[1]
        lmbd = self.alpha ** 2 * (dim + self.ket) - dim

        sigma_points = self.compute_sigma_points(states, covariances, lmbd, dim)
        c_weights, m_weights = self.compute_sigma_weights(lmbd, dim)

        expected_measurements = self.measurement_model(sigma_points)
        mean_measurement = np.sum(m_weights[:, np.newaxis] * sigma_points, axis=1)

        deviation = expected_measurements - mean_measurement[:, np.newaxis]
        scaled_deviation = c_weights[:, np.newaxis] * deviation
        innovation_covariances = np.sum(scaled_deviation[:, :, :, np.newaxis] @ deviation[:, :, np.newaxis],
                                        axis=1) + self.measurement_noise

        sigma_deviation = sigma_points - states[:, np.newaxis]
        scaled_sigma_deviation = c_weights[:, np.newaxis] * sigma_deviation
        kalman_factor = np.sum(scaled_sigma_deviation[:, :, :, np.newaxis] @ sigma_deviation[:, :, np.newaxis], axis=1)

        inv_inno = np.linalg.inv(innovation_covariances)
        kalman_gain = kalman_factor @ inv_inno
        return mean_measurement, innovation_covariances, inv_inno, kalman_gain

    def pure_update(self,
                    states,
                    covariances,
                    innovation,
                    innovation_covariances,
                    kalman_gain):
        updated_states = states + np.squeeze(innovation[:, np.newaxis] @ np.transpose(kalman_gain, (0, 2, 1)))
        updated_covariances = covariances - kalman_gain @ innovation_covariances @ np.transpose(kalman_gain, (0, 2, 1))
        return updated_states, updated_covariances
