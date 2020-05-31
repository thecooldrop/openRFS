"""
This module collects the Kalman filtering algorithms in one place. All of the implementations are subclasses of the
Kalman abstract base class.
"""

from abc import ABC, abstractmethod
from numbers import Number


class Kalman(ABC):
    """
    This class represent the base abstract class of all Kalman filters. It outlines the basic method which should
    be present in each Kalman filter, and defines the getter and setter properties which are needed for accesing
    fields by other algorithms
    """

    class Builder(ABC):
        """
        Builder class for abstract base Kalman filter defined as abstract class. It outlines the common initialization
        methods of Kalman filter builders.
        """

        def __init__(self):
            self._transition_model = None
            self._transition_noise = None
            self._measurement_model = None
            self._measurement_noise = None
            self._covariances = None
            self._states = None

        def with_transition_model(self, transition_model):
            """
            Initializes the transition model of Kalman filter instance

            This method expects the transition model to be np.ndarray of dimension two, but any value not None is
            accepted
            :param transition_model:
            :return: the builder instance
            """
            if transition_model is not None:
                self._transition_model = transition_model
            else:
                raise ValueError("The transition model for the Kalman filter can not be None")
            return self

        def with_transition_noise(self, transition_noise):
            """
            Initializes the transition noise of Kalman filter instance

            This method expects the transition model to be np.ndarray of dimension two, but any value not None is
            accepted
            :param transition_noise:
            :return: the builder instance
            """
            if transition_noise is not None:
                self._transition_noise = transition_noise
            else:
                raise ValueError("The transition noise for Kalman filter can not be None")
            return self

        def with_measurement_model(self, measurement_model):
            """
            Initializes the measurement model of Kalman filter instance

            This method expects the measurement model to be np.ndarray of dimension two, but any value not None is
            accepted
            :param measurement_model:
            :return: the builder instance
            """
            if measurement_model is not None:
                self._measurement_model = measurement_model
            else:
                raise ValueError("The measurement model for Kalman filter can not be None")
            return self

        def with_measurement_noise(self, measurement_noise):
            """
            Initializes the measurement noise of Kalman filter instance

            This method expects the measurement noise to be np.ndarray of dimension two, but any value not None is
            accepted
            :param measurement_noise:
            :return: the builder instance
            """
            if measurement_noise is not None:
                self._measurement_noise = measurement_noise
            else:
                raise ValueError("The measurement noise for Kalman filter can not be None")
            return self

        def with_states(self, states):
            """
            Initializes the initial states of Kalman filter instance

            This method expects the initial states to be np.ndarray of dimension two, but any value not None is
            accepted
            :param states:
            :return: the builder instance
            """
            if states is not None:
                self._states = states
            else:
                raise ValueError("The states of Kalman filter can not be None")
            return self

        def with_covariances(self, covariances):
            """
            Initializes the initial covariances of Kalman filter instance

            This method expects the initial covariances to be np.ndarray of dimension three, but any value not None is
            accepted
            :param covariances:
            :return: the builder instance
            """
            if covariances is not None:
                self._covariances = covariances
            else:
                raise ValueError("The covariances of Kalman filter can not be None")
            return self

        def _check_base_inputs_none(self):
            if self._transition_model is None:
                raise ValueError("The transition model of Kalman filter can not be None")
            if self._transition_noise is None:
                raise ValueError("The transition noise of Kalman filter can not be None")
            if self._measurement_model is None:
                raise ValueError("The measurement model of Kalman filter can not be None")
            if self._measurement_noise is None:
                raise ValueError("The measurement noise of Kalman filter can not be None")
            if self._states is None:
                raise ValueError("The initial state of the Kalman filter can not be None")
            if self._covariances is None:
                raise ValueError("The initial covariances of the Kalman filter can not be None")

        def _check_states_covariances_dimensions(self):
            if not self._covariances.ndim >= 2:
                raise ValueError("The initial covariances of Kalman filter have to have at least dimension of two")
            if self._states.ndim == 1:
                self._states = self._states[np.newaxis]
            if self._covariances.ndim == 2:
                self._covariances = self._covariances[np.newaxis]

        @abstractmethod
        def build(self):
            pass

    # TODO: Remove all non-mandatory parameters and initialize the fields properly
    @abstractmethod
    def __init__(self,
                 transition_model,
                 transition_noise,
                 measurement_model,
                 measurement_noise,
                 states=None,
                 covariances=None):
        self._transition_model = transition_model
        self._transition_noise = transition_noise
        self._measurement_model = measurement_model
        self._measurement_noise = measurement_noise
        self._states = states
        self._covariances = covariances
        self._expected_measurements = None
        self._innovations = None
        self._innovation_covariances = None
        self._inv_innovation_covariances = None
        self._kalman_gains = None

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
    def states(self):
        return self._states

    @property
    def covariances(self):
        return self._covariances

    @states.setter
    def states(self, value):
        self._states = value

    @covariances.setter
    def covariances(self, value):
        self._covariances = value

    @property
    def expected_measurements(self):
        return self._expected_measurements

    @property
    def innovations(self):
        return self._innovations

    @property
    def innovation_covariances(self):
        return self._innovation_covariances

    @property
    def inv_innovation_covariances(self):
        return self._inv_innovation_covariances


class KF(Kalman):
    class Builder(Kalman.Builder):
        def __init__(self):
            super(KF.Builder, self).__init__()

        def build(self):
            self._check_base_inputs_none()

            if not self._transition_model.ndim == 2:
                raise ValueError("The transition model has to be a two-dimensional matrix for Kalman filter")
            if not self._transition_noise.ndim == 2:
                raise ValueError("The transition noise has to be a two-dimensional matrix for Kalman filter")
            if not self._measurement_model.ndim == 2:
                raise ValueError("The measurement model for Kalman filter has to be a two-dimensional matrix")
            if not self._measurement_noise.ndim == 2:
                raise ValueError("The measurement noise for Kalman filter has to be a two-dimensional matrix")

            self._check_states_covariances_dimensions()

            # check compatibility between shapes of matrices
            states_shape = self._states.shape
            covariances_shape = self._covariances.shape
            transition_shape = self._transition_model.shape
            transition_noise_shape = self._transition_noise.shape
            measurement_shape = self._measurement_model.shape
            measurement_noise_shape = self._measurement_noise.shape

            if not states_shape[1] == covariances_shape[1]:
                error_string = "The shape of initial states and initial covariance matrices are incompatible. Number " \
                               "of rows of covariance matrix does not match the number of columns of the states "
                raise ValueError(error_string)

            if not states_shape[1] == covariances_shape[2]:
                error_string = "The shape of initial states and initial covariance matrices are incompatible. Number " \
                               "of columns of covariance matrix does not match the number of columns of the states "
                raise ValueError(error_string)

            if not states_shape[1] == transition_shape[1]:
                error_string = "The state vector can not be multiplied by transpose of transition model. The number " \
                               "of columns of transition model does not match the number of columns in the state vector"
                raise ValueError(error_string)

            if not transition_shape[0] == transition_noise_shape[0]:
                error_string = "The transition model matrix of shape " + str(transition_shape) + " can not be added" \
                               " to transition noise matrix of shape " + str(transition_noise_shape)
                raise ValueError(error_string)

            if not transition_shape[0] == transition_noise_shape[1]:
                error_string = "The transition model matrix of shape " + str(transition_shape) + " can not be added" \
                               " to transition noise matrix of shape " + str(transition_noise_shape)
                raise ValueError(error_string)

            if not states_shape[1] == measurement_shape[1]:
                error_string = "The state vector can not be multiplied by transpose of measurement model. The number " \
                               "of columns of measurement model does not match the number of columns in the state " \
                               "vector"
                raise ValueError(error_string)

            if not (measurement_shape[1] == covariances_shape[1] and measurement_shape[1] == covariances_shape[2]):
                raise ValueError("The measurement model and covariances have shape incompatible for computing the " \
                                 " measurement_model @ covariances @ measurement_model.T")

            if not (measurement_shape[0] == measurement_noise_shape[0] and measurement_shape[0] ==
                    measurement_noise_shape[1]):
                raise ValueError("The measurement model matrix and measurement noise matrix are not compatible for " \
                                 "addition.")

            return KF(np.copy(self._transition_model),
                      np.copy(self._transition_noise),
                      np.copy(self._measurement_model),
                      np.copy(self._measurement_noise),
                      np.copy(self._states),
                      np.copy(self._covariances))

    def __init__(self,
                 transition_model,
                 transition_noise,
                 measurement_model,
                 measurement_noise,
                 states=None,
                 covariances=None):

        super(KF, self).__init__(transition_model,
                                 transition_noise,
                                 measurement_model,
                                 measurement_noise,
                                 states,
                                 covariances)

    def predict(self):
        """
        Computes the predict step of classical Kalman filter

        :return: None

        Usage: This function can be used to compute prediction of many states at once. Limitation to
        vectorization is that all states are predicted with same transition model. It is also assumed that there is
        only single control vector which is applied to all predicted states.
        """

        self._states = self._states @ self._transition_model.T
        self._covariances = self._transition_model @ self._covariances @ self._transition_model.T + self._transition_noise

    def update(self, measurements):
        """
        Computes the update of each state with each measurement.

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

        :param measurements: Numpy array for storing measurements, where each row is treated as a measurement.

        :return: Returns updated states and covariance matrices
        """
        num_meas = measurements.shape[0]
        self._expected_measurements = self._states @ self._measurement_model.T
        self._innovation = measurements[:, np.newaxis, :] - self.expected_measurements
        self._innovation = np.transpose(self._innovation, (1, 0, 2))
        self.compute_update_matrices()
        self._covariances = np.repeat(self._covariances, num_meas, axis=0)
        self.pure_update()

    def compute_update_matrices(self):
        dim = self._states.shape[1]
        self._innovation_covariances = self._measurement_noise + \
                                       self._measurement_model @ self._covariances @ self._measurement_model.T
        self._inv_innovation_covariances = np.linalg.inv(self._innovation_covariances)
        self._kalman_gains = self._covariances @ self._measurement_model.T @ self._inv_innovation_covariances
        self._covariances = (np.eye(dim) - self._kalman_gains @ self._measurement_model) @ self._covariances

    def pure_update(self):
        self._states = self._states[:, np.newaxis, :] + self._innovation @ np.transpose(self._kalman_gains, (0, 2, 1))
        if self._states.size > 0:
            self._states = np.concatenate(self._states[:])
        else:
            self._states = np.empty((0, self._states.shape[-1]))


class EKF(Kalman):
    class Builder(Kalman.Builder):
        def __init__(self):
            super(EKF.Builder, self).__init__()
            self._transition_jacobi = None
            self._measurement_jacobi = None

        def with_measurement_model(self, measurement_model):
            if not callable(measurement_model):
                raise ValueError("The measurment model passed to EKF has to be callable")
            else:
                self._measurement_model = measurement_model
            return self

        def with_measurement_jacobi(self, measurement_jacobi):
            if not callable(measurement_jacobi):
                raise ValueError("The Jacobian of measurement model has to be a callable for EKF")
            else:
                self._measurement_jacobi = measurement_jacobi
            return self

        def with_transition_model(self, transition_model):
            if not callable(transition_model):
                raise ValueError("The transition model passed to EKF has to be callable")
            else:
                self._transition_model = transition_model
            return self

        def with_transition_jacobi(self, transition_jacobi):
            if not callable(transition_jacobi):
                raise ValueError("The Jacobian of transition model has to be callable for EKF")
            else:
                self._transition_jacobi = transition_jacobi
            return self

        def build(self):
            self._check_base_inputs_none()
            if self._transition_jacobi is None:
                raise ValueError("The transition Jacobian of the EKF can not be None")
            if self._measurement_jacobi is None:
                raise ValueError("The measurement Jacobian of the EKF can not be None")

            self._check_states_covariances_dimensions()
            states_shape = self._states.shape
            covariances_shape = self._covariances.shape
            if not states_shape[1] == covariances_shape[1]:
                error_string = "The shape of initial states and initial covariance matrices are incompatible. Number " \
                               "of rows of covariance matrix does not match the number of columns of the states "
                raise ValueError(error_string)

            if not states_shape[1] == covariances_shape[2]:
                error_string = "The shape of initial states and initial covariance matrices are incompatible. Number " \
                               "of columns of covariance matrix does not match the number of columns of the states "
                raise ValueError(error_string)

            return EKF(self._transition_model,
                       self._transition_jacobi,
                       np.copy(self._transition_noise),
                       self._measurement_model,
                       self._measurement_jacobi,
                       np.copy(self._measurement_noise),
                       np.copy(self._states),
                       np.copy(self._covariances))

    def __init__(self,
                 transition_model,
                 transition_jacobi,
                 transition_noise,
                 measurement_model,
                 measurement_jacobi,
                 measurement_noise,
                 states=None,
                 covariances=None):
        super(EKF, self).__init__(transition_model,
                                  transition_noise,
                                  measurement_model,
                                  measurement_noise,
                                  states,
                                  covariances)
        self._transition_jacobi = transition_jacobi
        self._measurement_jacobi = measurement_jacobi


    def predict(self):
        """
        :return: None
        """

        # jacobis is 3D matrix, since each of them is linearized about different state
        jacobis = self._transition_jacobi(self._states)
        self._states = self._transition_model(self._states)
        self._covariances = jacobis @ self._covariances @ jacobis.transpose((0, 2, 1)) + self._transition_noise

    def update(self, measurements):
        """
        Computes the update of each state with each measurement.

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
        :return: None
        """
        # ensure states is in row form
        num_meas = measurements.shape[0]
        self._expected_measurements = self._measurement_model(self._states)
        self._innovations = measurements[:, np.newaxis, :] - self.expected_measurements
        self._innovations = np.transpose(self._innovations, (1, 0, 2))
        self.compute_update_matrices()
        self._covariances = np.repeat(self._innovation_covariances, num_meas, axis=0)
        self.pure_update()

    def compute_update_matrices(self):
        dim = self._states.shape[1]
        jacobis = self._measurement_jacobi(self._states)
        self._innovation_covariances = self._measurement_noise + \
                                       jacobis @ self._covariances @ np.transpose(jacobis, (0, 2, 1))

        self._inv_innovation_covariances = np.linalg.inv(self._innovation_covariances)
        self._kalman_gains = self._covariances @ np.transpose(jacobis, (0, 2, 1)) @ self._inv_innovation_covariances
        self._covariances = (np.eye(dim) - self._kalman_gains @ jacobis) @ self._covariances

    def pure_update(self):
        self._states = self._states[:, np.newaxis, :] + self._innovation @ np.transpose(self._kalman_gains, (0, 2, 1))
        self._states = np.concatenate(self._states[:])


class UKF(Kalman):

    class Builder(EKF.Builder):

        def __init__(self):
            super(UKF.Builder, self).__init__()
            self._alpha = None
            self._beta = None
            self._ket = None

        def with_alpha(self, alpha):
            if not isinstance(alpha, Number):
                raise ValueError("The alpha parameter of UKF has to be a number")
            else:
                self._alpha = alpha
            return self

        def with_beta(self, beta):
            if not isinstance(beta, Number):
                raise ValueError("The beta parameter of UKF has to be a number")
            else:
                self._beta = beta
            return self

        def with_ket(self, ket):
            if not isinstance(ket, Number):
                raise ValueError("The ket parameter of UKF hast to be a number")
            else:
                self._ket = ket
            return self

        def build(self):
            self._check_base_inputs_none()
            if self._alpha is None:
                raise ValueError("The alpha parameter of UKF can not be None")
            if self._alpha < 0 :
                raise ValueError("The alpha parameter of UKF must be non-negative")
            if self._beta is None:
                raise ValueError("The beta parameter of UKF can not be None")
            if self._beta < 0:
                raise ValueError("The beta parameter of UKF has to be non-negative")
            if self._ket is None:
                raise ValueError("The ket parameter of UKF can not be None")
            if self._ket < 0:
                raise ValueError("The ket parameter of UKF has to be non-negative")
            if self._transition_jacobi is None:
                raise ValueError("The transition Jacobian of UKF can not be None")
            if self._measurement_jacobi is None:
                raise ValueError("The measurement Jacobian of UKF can not be None")
            self._check_states_covariances_dimensions()




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
                 covariances=None):
        super(UKF, self).__init__(transition_model,
                                  transition_noise,
                                  measurement_model,
                                  measurement_noise,
                                  states,
                                  covariances)

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
        self._innovations = measurements[:, np.newaxis, :] - self._predicted_mean_measurement
        self._innovations = np.transpose(self._innovations, (1, 0, 2))
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
        self._expected_measurements = self._predicted_mean_measurement
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


if __name__ == "__main__":
    import numpy as np
    from timeit import timeit
    from estimation.kalman_factory import ccv_model
    import cProfile

    transition_matrix, transition_noise = ccv_model(1 / 30, 5, 4)
    meas_matrix = np.eye(8)
    meas_noise_matrix = np.eye(8)
    initial_state = np.random.random((50, 8))
    init_cov = np.random.random((50, 8, 8))
    init_cov = 1/2*(init_cov + np.transpose(init_cov, (0, 2, 1))) + 5*np.eye(8)
    def test_speed():
        for _ in range(10000):
            kf_filter = KF.Builder().with_states(initial_state) \
                                    .with_covariances(init_cov) \
                                    .with_transition_model(transition_matrix) \
                                    .with_transition_noise(transition_noise) \
                                    .with_measurement_model(meas_matrix) \
                                    .with_measurement_noise(meas_noise_matrix) \
                                    .build()
            measurements = np.random.random((100,8))
            kf_filter.predict()
            kf_filter.update(measurements)

    cProfile.run('test_speed()', sort='cumtime')

