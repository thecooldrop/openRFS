import numpy as np
from typing import Optional, Callable


def predict(states: np.ndarray,
            covariances: np.ndarray,
            transition: Callable,
            transition_jacobi : Callable,
            process_covariance: np.ndarray,
            control: Optional[np.ndarray] = None,
            control_transition: Optional[np.ndarray] = None):
    """
    Computes the predict step of extended Kalman filter

    :param states: Numpy array storing N states, where each row represents a state

    :param covariances: Numpy array storing covariances matching to states in NxMxM format where M is dimensionality
                        of state vector

    :param transition:  A function computing the transition for matrix of states

    :param process_covariance: Numpy array of shape MxM or NxMxM if each state has different noise

    :param control: Numpy array for storing either a single control vector or MxN matrix storing N control vectors,
                    one for each of the states

    :param control_transition: Transition matrices for control vectors

    :return: Predicted states and covariances
    """

    # ensure that states is in row form
    states = np.atleast_2d(states)
    dim = states.shape[1]

    # initialize the control vector or ensure it is in row form
    if not control:
        control = np.zeros((1, dim))
    else:
        control = np.atleast_2d(control)

    # initialize control transition or ensure that it is in form for vectorized operations
    if not control_transition:
        control_transition = np.eye(dim)

    # jacobis is 3D matrix, since each of them is linearized about different state
    jacobis = np.atleast_3d(transition_jacobi(states))
    predicted_states = transition(states) + control @ control_transition.T
    predicted_covariances = jacobis @ covariances @ jacobis.transpose((0,2,1)) + process_covariance

    return predicted_states, predicted_covariances


def update(states: np.ndarray,
           covariances: np.ndarray,
           measurement_function: Callable,
           measurement_jacobi: Callable,
           measurement_noise_covariance: np.ndarray,
           measurements: np.ndarray):
    """

    :param states: Numpy array storing N states, where each row represents a state
    :param covariances: Numpy array storing covariances matching to states in NxMxM format where M is dimensionality
                        of state vector
    :param measurement_function:
    :param measurement_jacobi:
    :param measurement_noise_covariance:
    :param measurements:
    :return:
    """
    # ensure states is in row form
    states = np.atleast_2d(states)
    dim = state.shape[1]
    expected_measurements = np.atleast_2d(measurement_function(states))
    jacobis = np.atleast_3d(measurement_jacobi(states))
    innovation = measurements - expected_measurements
    innovation_covariance =  measurement_noise_covariance + jacobis @ covariances @ np.transpose(jacobis, (0,2,1))
    inv_innovation_covariance = np.linalg.inv(innovation_covariance)
    kalman_gain = covariances @ np.transpose(jacobis, (0,2,1)) @ inv_innovation_covariance
    updated_states = states + innovation @ np.transpose(kalman_gain, (0,2,1))
    updated_covariances = (np.eye(dim) - kalman_gain @ jacobis) @ covariances
    return updated_states, updated_covariances

