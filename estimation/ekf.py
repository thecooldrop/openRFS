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
    Computes the predict step of classical Kalman filter

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
    if states.ndim > 1:
        dim = states.shape[1]
    else:
        dim = states.shape[0]
        states = states[np.newaxis]

    # initialize the control vector or ensure it is in row form
    if not control:
        control = np.zeros((1, dim))
    else:
        control = control.reshape((1, dim))

    # initialize control transition or ensure that it is in form for vectorized operatiosn
    if not control_transition:
        control_transition = np.eye(dim)
    else:
        control_transition = control_transition.reshape((dim,dim))

    # jacobis is 3D matrix, since each of them is linearized about different state
    jacobis = transition_jacobi(states)
    predicted_states = transition(states) + control.T @ control_transition.T
    predicted_covariances = jacobis @ covariances @ jacobis.transpose((0,2,1)) + process_covariance

    return predicted_states, predicted_covariances


def update(states: np.ndarray,
           covariances: np.ndarray,
           measurement_function: Callable,
           measurement_jacobi: Callable,
           measurement_noise_covariance: np.ndarray,
           measurements: np.ndarray):
    """

    :param states:
    :param covariances:
    :param measurement_function:
    :param measurement_jacobi:
    :param measurement_noise_covariance:
    :param measurements:
    :return:
    """

    if states.ndim > 1:
        dim_states = states.shape[1]
    else:
        dim_states = states.shape[0]
        states = states.reshape((1,dim_states))

    expected_measurements = measurement_function(states)
    jacobis = measurement_jacobi(states)

    innovation = expected_measurements - states @ jacobis.T
