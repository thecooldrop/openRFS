from abc import ABC, abstractmethod


class Kalman(ABC):
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def update(self, measurements):
        pass
