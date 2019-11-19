from abc import ABC, abstractmethod


class Kalman(ABC):

    @abstractmethod
    def __init__(self,
                 transition_model,
                 transition_noise,
                 measurement_model,
                 measurement_noise,
                 states=None,
                 covariances=None,
                 innovation=None,
                 innovation_covariances=None,
                 inv_innovation_covariances=None,
                 kalman_gains=None):
        self._transition_model = transition_model
        self._transition_noise = transition_noise
        self._measurement_model = measurement_model
        self._measurement_noise = measurement_noise
        self._innovation = innovation
        self._innovation_covariances = innovation_covariances
        self._inv_innovation_covariances = inv_innovation_covariances
        self._kalman_gains = kalman_gains
        self._states = states
        self._covariances = covariances

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, measurements):
        pass

    @abstractmethod
    def pure_update(self):
        pass

    @abstractmethod
    def compute_update_matrices(self):
        pass

    @property
    def state(self):
        return self._states

    @property
    def covariances(self):
        return self._covariances

    @property
    def innovation_covariances(self):
        return self._innovation_covariances

    @property
    def inv_innovation_covariances(self):
        return self._inv_innovation_covariances

    @state.setter
    def state(self, value):
        self._states = value

    @covariances.setter
    def covariances(self, value):
        self._covariances = value