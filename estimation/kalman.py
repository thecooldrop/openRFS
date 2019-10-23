from abc import ABC, abstractmethod


class Kalman(ABC):
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def compute_update_matrices(self):
        pass

    @abstractmethod
    def pure_update(self):
        pass
