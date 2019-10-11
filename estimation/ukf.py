import numpy as np
from typing import  Callable


def predict(states: np.ndarray,
            covariances: np.ndarray,
            transition: Callable,
            process_covariance: np.ndarray,
            alpha: float,
            beta: float,
            ket: float):
    """

    :param beta:
    :param states:
    :param covariances:
    :param transition:
    :param process_covariance:
    :param alpha:
    :param ket:
    :return:

    Usage:
    Note that alpha can also be an array, so that each
    """

    dim = states.shape[1]
    lmbd = alpha ** 2 * (dim + ket) - dim

    sigma_points = compute_sigma_points(states, covariances, lmbd, dim)
    c_weights, m_weights = compute_sigma_weights(alpha, beta, lmbd, dim)

    predictions = transition(sigma_points)
    predicted_state = np.sum(m_weights[:, np.newaxis] * predictions, axis=1)
    deviation = predictions - predicted_state[:, np.newaxis]
    scaled_deviation = c_weights[:, np.newaxis] * deviation
    predicted_covariance = np.sum(scaled_deviation[:, :, :, np.newaxis] @ deviation[:, :, np.newaxis],
                                  axis=1) + process_covariance
    return predicted_state, predicted_covariance


def update(states,
           covariances,
           measurement_model,
           measurement_noise,
           alpha,
           beta,
           ket,
           measurements):
    dim = states.shape[1]
    lmbd = alpha ** 2 * (dim + ket) - dim

    sigma_points = compute_sigma_points(states, covariances, lmbd, dim)
    c_weights, m_weights = compute_sigma_weights(alpha, beta, lmbd, dim)

    expected_measurements = measurement_model(sigma_points)
    mean_measurement = np.sum(m_weights[:, np.newaxis] * sigma_points, axis=1)

    deviation = expected_measurements - mean_measurement[:, np.newaxis]
    scaled_deviation = c_weights[:, np.newaxis] * deviation
    innovation_covariances = np.sum(scaled_deviation[:, :, :, np.newaxis] @ deviation[:, :, np.newaxis],
                                    axis=1) + measurement_noise

    sigma_deviation = sigma_points - states[:, np.newaxis]
    scaled_sigma_deviation = c_weights[:, np.newaxis] * sigma_deviation
    kalman_factor = np.sum(scaled_sigma_deviation[:, :, :, np.newaxis] @ sigma_deviation[:, :, np.newaxis], axis=1)

    kalman_gain = kalman_factor @ np.linalg.inv(innovation_covariances)
    innovation = measurements[:, np.newaxis, :] - mean_measurement
    updated_states = states[:, np.newaxis] + innovation @ np.transpose(kalman_gain, (0, 2, 1))
    updated_covariances = covariances - kalman_gain @ innovation_covariances @ np.transpose(kalman_gain, (0, 2, 1))
    return np.squeeze(updated_states), updated_covariances


def compute_sigma_weights(alpha, beta, lmbd, n):
    weight: float = lmbd / (n + lmbd)
    m_weights = np.repeat(weight, 2 * n + 1)
    c_weights = np.repeat(weight, 2 * n + 1)
    c_weights[0] = c_weights[0] + (1 - alpha ** 2 + beta)
    c_weights[1:] = (1 / (2 * lmbd)) * c_weights[1:]
    m_weights[1:] = (1 / (2 * lmbd)) * m_weights[1:]
    return c_weights, m_weights


def compute_sigma_points(states, covariances, lmbd, n):
    root_covariances = np.linalg.cholesky((lmbd + n) * covariances)
    # To each state we compute a matrix of sigma points
    sig_first = states[:, np.newaxis] + np.transpose(root_covariances, (0, 2, 1))
    sig_second = states[:, np.newaxis] - np.transpose(root_covariances, (0, 2, 1))
    sigma_points = np.concatenate((states[:, np.newaxis], sig_first, sig_second), axis=1)
    return sigma_points
