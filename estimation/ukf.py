import numpy as np
from estimation.kalman import Kalman


class UKF(Kalman):

    def __init__(self,
                 transition_model,
                 transition_jacobi,
                 transition_noise,
                 measurement_model,
                 measurement_jacobi,
                 measurement_noise,
                 alpha,
                 beta,
                 ket,
                 states=None,
                 covariances=None,
                 innovation=None,
                 innovation_covariances=None,
                 inv_innovation_covariances=None,
                 kalman_gains=None,
                 ):
        super(UKF, self).__init__(transition_model,
                                  transition_noise,
                                  measurement_model,
                                  measurement_noise,
                                  states,
                                  covariances,
                                  innovation,
                                  innovation_covariances,
                                  inv_innovation_covariances,
                                  kalman_gains)
        self._transition_jacobi = transition_jacobi
        self._measurement_jacobi = measurement_jacobi
        self._alpha = alpha
        self._beta = beta
        self._ket = ket
        self._predicted_mean_measurement = None

    def predict(self):
        """
        Computes the UKF prediction step for non-linear motion model.
        :return: none
        """
        dim = self._states.shape[1]
        lmbd = self._alpha ** 2 * (dim + self._ket) - dim

        sigma_points = self._compute_sigma_points(lmbd, dim)
        c_weights, m_weights = self._compute_sigma_weights(lmbd, dim)

        # note that predictions is a three dimensional matrix
        # Entries along first dimension correspond to prior states
        # Entries along second and third dimension correspond to sigma points
        predictions = self._transition_model(sigma_points)

        # multiply n-th row of each sigma points matrix by n-th weight and sum the rows to obtain the predicted states
        self._states = np.sum(m_weights[:, np.newaxis] * predictions, axis=1)

        # subtract from each prediction matrix the matching state by rearranging the states into matrices
        deviation = predictions - self._states[:, np.newaxis]

        # multiply n-th row of each matrix stored along the first dimension by n-th weight
        scaled_deviation = c_weights[:, np.newaxis] * deviation

        # observe the rows of scaled_deviation and deviation
        # Compute the sum of outer product of rows in scaled_deviation and deviation
        self._covariances = self._outer_sum_product(scaled_deviation, deviation) + self._transition_noise

    def update(self, measurements):
        """
        Computes the UKF update of each state with each measurement

        If before the method there were 10x4 states and measurements are 5x3 then after executing this method following
        matrices are stored:
            - _innovation = 10x5x3 where first dimension is indexed by old state and second by measurement, where
            objects under consideration are rows

            - _innovation_covariances = 10x3x3 where the first dimension is indexed by old state and the objects of
            interest are 3x3 matrices

            - _inv_innovation_covarainces = 10x3x3 where the first dimension is indexed by old state and objects of
            interest are 3x3 matrices

            - _kalman_gains = 10x4x3, where the first dimension is indexed by old state and objects of interest are
            4x3 matrices

            - _covariances = 50x4x4 where the first dimension is indexed by new state. The order of elements in this
            matrix is such that first five 4x4 matrices represent the results of updating the first old state with
            each of five measurements, and so on.

            - _states = 50x4 where the first dimension is indexed by new state. The order of elements in this matrix
            is such that first five rows are states which result from updating the first state with all of five
            measurements

            - _innovation = 10x5x3 where the first dimension is indexed by old state and second by measurement. This
            means that first 5x3 matrix contains the rows which result from subtracting each of measurements from the
            first old state and so on.
        :param measurements:
        :return:
        """
        num_meas = measurements.shape[0]
        self.compute_update_matrices()
        # compute the difference between each measurement and each predicted mean measurement
        # and transpose so that n-th entry along first dimension of self._innovation contains the
        # difference all of measurements and the n-th state
        self._innovation = measurements[:, np.newaxis, :] - self._predicted_mean_measurement
        self._innovation = np.transpose(self._innovation, (1, 0, 2))
        self.pure_update()
        self._covariances = np.repeat(self._covariances, num_meas, axis=0)

    def compute_update_matrices(self):
        dim = self._states.shape[1]
        lmbd = self._alpha ** 2 * (dim + self._ket) - dim

        sigma_points = self._compute_sigma_points(lmbd, dim)
        c_weights, m_weights = self._compute_sigma_weights(lmbd, dim)

        # Note that expected_measurements is three dimensional matrix
        # The n-th entry along first dimension corresponds to measurements generated by sigma points of n-th state
        # Note that expected measurements is two dimensional and the n-th row corresponds to expected measurement of
        # n-th state vector
        expected_measurements = self._measurement_model(sigma_points)

        # Multiply n-th row of each matrix stored along first dimension by n-th weight
        # This computes the predicted mean measurement for each state.
        # n-th row contains the predicted mean measurement for n-th predicted state
        self._predicted_mean_measurement = np.sum(m_weights[:, np.newaxis] * expected_measurements, axis=1)

        # subtract the predicted mean measurement from measurement sigma points for matching state
        deviation = expected_measurements - self._predicted_mean_measurement[:, np.newaxis]
        scaled_deviation = c_weights[:, np.newaxis] * deviation
        self._innovation_covariances = self._outer_sum_product(scaled_deviation, deviation) + self._measurement_noise

        # note that sigma_points is three dimensional
        sigma_deviation = sigma_points - self._states[:, np.newaxis]
        scaled_sigma_deviation = c_weights[:, np.newaxis] * sigma_deviation
        kalman_factor = self._outer_sum_product(scaled_sigma_deviation, deviation)

        inv_inno = np.linalg.inv(self._innovation_covariances)
        self._inv_innovation_covariances = inv_inno
        self._kalman_gains = kalman_factor @ inv_inno

    def pure_update(self):

        # note that self._innovation is 3D matrix
        self._states = self._states[:, np.newaxis, :] + self._innovation @ np.transpose(self._kalman_gains, (0, 2, 1))
        self._states = np.concatenate(self._states[:])
        self._covariances = self._covariances - \
                            self._kalman_gains @ self._innovation_covariances @ np.transpose(self._kalman_gains,
                                                                                             (0, 2, 1))

    def _compute_sigma_weights(self, lmbd, n):
        """
        Computes the weights which are used for predicting the state.
        :param lmbd: UKF parameter
        :param n: dimensionality of state vector
        :return: c_weights is a 1-D vector of length (2n+1)
        :return: m_weights is a 1-D vector of length (2n+1)
        """
        weight: float = lmbd / (n + lmbd)
        m_weights = np.repeat(weight, 2 * n + 1)
        c_weights = np.repeat(weight, 2 * n + 1)
        c_weights[0] = c_weights[0] + (1 - self._alpha ** 2 + self._beta)
        c_weights[1:] = (1 / (2 * lmbd)) * c_weights[1:]
        m_weights[1:] = (1 / (2 * lmbd)) * m_weights[1:]
        return c_weights, m_weights

    def _compute_sigma_points(self, lmbd, n):
        """
        Computes the sigma points for states and covariances.

        Given that states and covariances have shapes 10x4 and 10x4x4 respectively the resulting sigma points has
        dimension of 10x9x4. Here each 9x4 matrix represents the sigma points of each respective state
        :param lmbd: The paramater of UKF filter
        :param n: Dimensionality of state vector
        :return: Sigma points of each state of shape lx(2n+1)xn where l is number of states available from previous step
        """
        root_covariances = np.linalg.cholesky((lmbd + n) * self._covariances)
        # To each state we compute a matrix of sigma points
        sig_first = self._states[:, np.newaxis] + np.transpose(root_covariances, (0, 2, 1))
        sig_second = self._states[:, np.newaxis] - np.transpose(root_covariances, (0, 2, 1))
        sigma_points = np.concatenate((self._states[:, np.newaxis], sig_first, sig_second), axis=1)
        return sigma_points

    def _outer_sum_product(self, fst, snd):
        """
        Computes the sum of outer products between rows of first and second argument

        If fst has dimension mxn and snd has dimension mxp then output has
        """
        # arrange the rows of A into columns of expanded matrix
        expanded_a = np.expand_dims(fst, axis=-1)

        # arrange the rows of B into rows by adding a dimension
        expanded_b = np.expand_dims(snd, axis=-2)

        # compute the outer products as simple matrix product
        outer_product = expanded_a @ expanded_b

        # if number of dimensions is two there is nothing to sum so
        # simply return the product and otherwise compute the sum of outer products
        if outer_product.ndim == 2:
            return outer_product
        else:
            return np.sum(outer_product, axis=-3)
