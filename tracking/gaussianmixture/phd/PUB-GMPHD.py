from util.mixins import GaussianMixtureMixin
import numpy as np

class PUB_GMPHD(GaussianMixtureMixin):

    class Builder():
        def __init__(self):
            pass

    def __init__(self,
                 weights,
                 means,
                 covariances,
                 birth_weights,
                 birth_means,
                 birth_covariances,
                 probability_detection,
                 probability_survival,
                 clutter_density,
                 kalman_filter):

        super(PUB_GMPHD, self).__init__()
        self._weights = weights
        self._means = means
        self._covariances = covariances
        self._birth_weights = birth_weights
        self._birth_means = birth_means
        self._birth_covariances = birth_covariances
        self._probability_detection = probability_detection
        self._probability_survival = probability_survival
        self._clutter_density = clutter_density
        self._kalman_filter = kalman_filter


    def process(self, measurements):
        if self._weights.size > 0:
            self._kalman_filter.states = self._means
            self._kalman_filter.covariances = self._covariances
        else:
            self._kalman_filter.states = np.zeros((0, self._birth_means.shape[1]))
            self._kalman_filter.covariances = np.zeros((0, self._birth_covariances.shape[1], self._birth_covariances.shape[2]))
        predicted_weights, predicted_means, predicted_covariances = self._predict_existing_components()



    def _predict_existing_components(self):
        weights = self._probability_survival * self._weights
        self._kalman_filter.predict()
        return weights, self._kalman_filter.states, self._kalman_filter.covariances

    def _update_existing_components(self, ):
