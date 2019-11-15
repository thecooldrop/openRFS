from abc import ABC, abstractmethod


class Kalman(ABC):
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, measurements):
        pass

    @abstractmethod
    def compute_update_matrices(self):
        pass

    @abstractmethod
    def pure_update(self, innovations, kalman_gains):
        pass
