import numpy as np
from typing import Optional
from estimation.kalman import Kalman


class KF(Kalman):

    def __init__(self, transition_model, transition_noise, measurement_model, measurement_noise):
        self.transition_model = transition_model
        self.transition_noise = transition_noise
        self.measurement_model = measurement_model
        self.measurement_noise = measurement_noise

    def predict(self,
                states: np.ndarray,
                covariances: np.ndarray,
                control: Optional[np.ndarray] = None,
                control_transition: Optional[np.ndarray] = None):
        """
        Computes the predict step of classical Kalman filter

        :param states: Numpy array storing N states, where each row represents a state

        :param covariances: Numpy array storing covariances matching to states in NxMxM format where M is dimensionality
                            of state vector

        :param control: Numpy array for storing either a single control vector in row form

        :param control_transition: Transition matrices for control vectors

        :return: Predicted states and covariances
        """

        # ensure that states is in row form
        dim = states.shape[1]

        # initialize the control vector or ensure it is in row form
        if not control:
            control = np.zeros((1, dim))
        else:
            control = np.atleast_2d(control)

        # initialize control transition or ensure that it is in form for vectorized operations
        if not control_transition:
            control_transition = np.eye(dim)

        predicted_states = states @ self.transition_model.T + control @ control_transition.T
        predicted_covariances = self.transition_model @ covariances @ self.transition_model.T + self.transition_noise

        return predicted_states, predicted_covariances

    def update(self,
               states: np.ndarray,
               covariances: np.ndarray,
               measurements: np.ndarray):
        """

        :param states: Numpy array storing N states, where each row represents a state

        :param covariances: Numpy array storing covariances matching to states in NxMxM format where M is dimensionality
                            of state vector

        :param measurements: Numpy array for storing measurements, where each row is treated as a measurement.

        :return: Returns updated states and covariance matrices
        """

        # matrix times vector operations are transposed because the states and measurements are viewed as stored in rows
        if self.measurement_noise.ndim < 3:
            self.measurement_noise = self.measurement_noise[np.newaxis]

        innovation = measurements - states @ self.measurement_model.T

        _, _, kalman_gain = self.compute_update_matrices(covariances)

        return self.pure_update(states, covariances, innovation, kalman_gain)

    def compute_update_matrices(self,
                                covariances):

        innovation_covariance = self.measurement_noise + self.measurement_model @ covariances @ self.measurement_model.T
        inv_innovation_covariances = np.linalg.inv(innovation_covariance)
        kalman_gain = covariances @ self.measurement_model.T @ inv_innovation_covariances
        return innovation_covariance, inv_innovation_covariances, kalman_gain

    def pure_update(self,
                    states,
                    covariances,
                    innovation,
                    kalman_gain):

        dim = states.shape[1]
        updated_states = states + np.squeeze(innovation[:, np.newaxis] @ np.transpose(kalman_gain, (0, 2, 1)))
        updated_covariances = (np.eye(dim) - kalman_gain @ self.measurement_model) @ covariances
        return updated_states, updated_covariances
