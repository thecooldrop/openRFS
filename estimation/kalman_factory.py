import numpy as np
from estimation.kalman import KF, EKF, UKF
from abc import ABC, abstractmethod
from enum import Enum, auto


def ccv_model(timestep, acceleration_variance, num_vels):
    """
    This method returns matrices for transition model and transition noise for continuous constant velocity model.

    The method returns two values where both matrices contain (2n)x(2n) elements. The first returned matrix represents
    the transition matrix, while the second matrix represents the noise matrix.

    For detailed derivation of the model refer to chapter 6.2.2 of book:
    Estimation with Applications to Tracking and Navigation by Yaakov Bar-Shalom, X. Rong Li and
    Thiagalingam Kirubarajan

    :param timestep: the time which passes between two measurements
    :param acceleration_variance: the variance of acceleration in constant velocity model
    :param num_vels: number of velocity components, denoted by n in doc
    :return: two matrices representing the transition model matrix and noise matrix
    """
    t = timestep
    kron_core = np.eye(num_vels)
    transition_core = np.reshape([1, t, 0, 1], (2, 2))
    transition_matrix = np.kron(kron_core, transition_core)

    noise_core = np.reshape([1 / 3 * t ** 3, 1 / 2 * t ** 2, 1 / 2 * t ** 2, t], (2, 2))
    noise_matrix = np.kron(kron_core, noise_core) * acceleration_variance

    return transition_matrix, noise_matrix


def cca_model(timestep, jerk_variance, num_vels):
    """
    This method returns for transition model and transition noise matrices for continuous constant acceleration model

    For detailed derivation of the model refer to chapter 6.2.3 of book:
    Estimation with Applications to Tracking and Navigation by Yaakov Bar-Shalom, X. Rong Li and
    Thiagalingam Kirubarajan

    :param timestep: the time which passes between two measurements
    :param jerk_variance: the variance of jerk in constant acceleration model
    :param num_vels: number of velocity components
    :return: two matrices represening transition model matrix and noise matrix
    """
    t = timestep
    kron_core = np.eye(num_vels)
    transition_core = np.reshape([1, t, 1 / 2 * t ** 2, 0, 1, t, 0, 0, 1], (3, 3))
    transition_matrix = np.kron(kron_core, transition_core)

    noise_core = np.reshape(
        [1 / 20 * t ** 5, 1 / 8 * t ** 4, 1 / 6 * t ** 3, 1 / 8 * t ** 4, 1 / 3 * t ** 3, 1 / 2 * t ** 2,
         1 / 6 * t ** 3, 1 / 2 * t ** 2, t], (3, 3)) * jerk_variance
    noise_matrix = np.kron(kron_core, noise_core)

    return transition_matrix, noise_matrix


def dwna_model(timestep, acceleration_variance, num_vels):
    """
    Returns transition matrix and transition variance matrix for discrete white noise acceleration model

    For more detailed information about the model refer to chapter 6.3.2 of book:
    Estimation with Applications to Tracking and Navigation by Yaakov Bar-Shalom, X. Rong Li and
    Thiagalingam Kirubarajan

    :param timestep: time which elapses between two measurements
    :param acceleration_variance: variance of acceleration
    :param num_vels: number of velocity components
    :return: transition model matrix and transition nose matrix
    """

    t = timestep
    kron_core = np.eye(num_vels)
    transition_core = np.reshape([1, t, 0, 1], (2, 2))
    transition_matrix = np.kron(kron_core, transition_core)

    noise_core = np.reshape([1 / 4 * t ** 4, 1 / 2 * t ** 3, 1 / 2 * t ** 3, t ** 2], (2, 2)) * acceleration_variance
    noise_matrix = np.kron(kron_core, noise_core)
    return transition_matrix, noise_matrix


def dwpa_model(timestep, jerk_variance, num_vels):
    """
    Returns the transition matrix and transition variance matrix for discrete Wiener process acceleration model

    For more detailed information about model and refer to chapter 6.3.3 of book:
    Estimation with Applications to Tracking and Navigation by Yaakov Bar-Shalom, X. Rong Li and
    Thiagalingam Kirubarajan

    :param timestep: time which elapses between two measurements
    :param jerk_variance: the variance of discrete jerk
    :param num_vels: number of velocity components
    :return: transition model matrix and noise convariance matrix
    """

    t = timestep
    kron_core = np.eye(num_vels)
    transition_core = np.reshape([1, t, 1 / 2 * t ** 2, 0, 1, t, 0, 0, 1], (3, 3))
    transition_matrix = np.kron(kron_core, transition_core)

    noise_core = np.reshape(
        [1 / 4 * t ** 4, 1 / 2 * t ** 3, 1 / 2 * t ** 2, 1 * 2 * t ** 3, t ** 2, t, 1 / 2 * t ** 2, t, 1],
        (3, 3)) * jerk_variance
    noise_matrix = np.kron(kron_core, noise_core)
    return transition_matrix, noise_matrix