import numpy as np
from typing import Optional


def predict(states: np.ndarray,
            covariances: np.ndarray,
            transition: np.ndarray,
            process_covariance: np.ndarray,
            control: Optional[np.ndarray] = None,
            control_transition: Optional[np.ndarray] = None):
    """
    Computes the predict step of classical Kalman filter

    :param states: Numpy array storing N states, where each row represents a state

    :param covariances: Numpy array storing covariances matching to states in NxMxM format where M is dimensionality
                        of state vector

    :param transition:  Numpy array of shape MxM

    :param process_covariance: Numpy array of shape MxM

    :param control: Numpy array for storing single control vector in row form

    :param control_transition: Transition matrices for control vectors

    :return: Predicted states and covariances

    Usage:
    This function can be used to create compute prediction of many states at once. Limitation to vectorization is that
    all states are predicted with same transition model. It is also assumed that there is only single control vector
    which is applied to all predicted states.
    """

    # ensure that states is in row form
    states = np.atleast_2d(states);
    dim = states.shape[1];

    # initialize the control vector or ensure it is in row form
    if not control:
        control = np.zeros((1, dim))
    else:
        control = np.atleast_2d(control)

    # initialize control transition or ensure that it is in form for vectorized operations
    if not control_transition:
        control_transition = np.eye(dim)

    predicted_states = states @ transition.T + control @ control_transition.T
    predicted_covariances = transition @ covariances @ transition.T + process_covariance

    return predicted_states, predicted_covariances


def update(states: np.ndarray,
           covariances: np.ndarray,
           measurement_matrix: np.ndarray,
           measurement_noise_covariance: np.ndarray,
           measurements: np.ndarray):
    """

    :param states: Numpy array storing N states, where each row represents a state

    :param covariances: Numpy array storing covariances matching to states in NxMxM format where M is dimensionality
                        of state vector

    :param measurement_matrix: Numpy array storing the linear measurement matrix or stack of matrices if each state
                               needs to be measured with different matrix

    :param measurement_noise_covariance: Numpy array for storing the measurement covariance matrix or stack of
                                         measurement covariance matrices if each of the states has different noise

    :param measurements: Numpy array for storing measurements, where each row is treated as a measurement.

    :return: Returns updated states and covariance matrices
    """

    if states.ndim > 1:
        dim = states.shape[1]
    else:
        dim = states.shape[0]

    # matrix times vector operations are transposed because the states and measurements are viewed as stored in rows
    if measurement_matrix.ndim < 3:
        measurement_matrix = measurement_matrix[np.newaxis]

    innovation = measurements - states @ measurement_matrix.T
    innovation_covariance = measurement_noise_covariance + measurement_matrix @ covariances @ measurement_matrix.T
    inv_innovation_covariances = np.linalg.inv(innovation_covariance)
    kalman_gain = covariances @ measurement_matrix.T @ inv_innovation_covariances
    updated_states = states + innovation @ kalman_gain.T
    updated_covariances = (np.eye(dim) - kalman_gain @ measurement_matrix) @ covariances
    return updated_states, updated_covariances
