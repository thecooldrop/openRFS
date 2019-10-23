import numpy as np
from typing import Optional
from estimation.kalman import Kalman


class EKF(Kalman):

    def __init__(self, transition_model, transition_jacobi, transition_noise,
                 measurement_model, measurement_jacobi, measurement_noise):
        self.transition_model = transition_model
        self.transition_noise = transition_noise
        self.transition_jacobi = transition_jacobi
        self.measurement_model = measurement_model
        self.measurement_noise = measurement_noise
        self.measurement_jacobi = measurement_jacobi

    def predict(self,
                states: np.ndarray,
                covariances: np.ndarray,
                control: Optional[np.ndarray] = None,
                control_transition: Optional[np.ndarray] = None):
        """
        Computes the predict step of extended Kalman filter

        :param states: Numpy array storing N states, where each row represents a state

        :param covariances: Numpy array storing covariances matching to states in NxMxM format where M is dimensionality
                            of state vector

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
        jacobis = self.transition_jacobi(states)
        predicted_states = self.transition_model(states) + control @ control_transition.T
        predicted_covariances = jacobis @ covariances @ jacobis.transpose((0, 2, 1)) + self.transition_noise

        return predicted_states, predicted_covariances

    def update(self,
               states: np.ndarray,
               covariances: np.ndarray,
               measurements: np.ndarray):
        """

        :param states: Numpy array storing N states, where each row represents a state
        :param covariances: Numpy array storing covariances matching to states in NxMxM format where M is dimensionality
                            of state vector
        :param measurements:
        :return:
        """
        # ensure states is in row form
        states = np.atleast_2d(states)
        expected_measurements = self.measurement_model(states)
        innovation = measurements - expected_measurements

        jacobis, _, _, kalman_gain = self.compute_update_matrices(states,
                                                                  covariances)

        return self.pure_update(states, covariances, innovation, jacobis, kalman_gain)

    def compute_update_matrices(self,
                                states,
                                covariances):

        jacobis = self.measurement_jacobi(states)
        innovation_covariance = self.measurement_noise + jacobis @ covariances @ np.transpose(jacobis, (0, 2, 1))
        inv_innovation_covariance = np.linalg.inv(innovation_covariance)
        kalman_gain = covariances @ np.transpose(jacobis, (0, 2, 1)) @ inv_innovation_covariance
        return jacobis, innovation_covariance, inv_innovation_covariance, kalman_gain

    def pure_update(self,
                    states,
                    covariances,
                    innovation,
                    jacobis,
                    kalman_gain):

        dim = states.shape[1]
        updated_states = states + np.squeeze(innovation[:, np.newaxis] @ np.transpose(kalman_gain, (0, 2, 1)))
        updated_covariances = (np.eye(dim) - kalman_gain @ jacobis) @ covariances
        return updated_states, updated_covariances
