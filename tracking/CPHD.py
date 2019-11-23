import numpy as np
import scipy.special as spec
from estimation.kalman import Kalman


class CPHD:
    """
    This class is an implementation of CPHD Gaussian mixture filter as described in following paper:

    Analytic Implementations of the Cardinalized Probability Hypothesis Density Filter - Ba-Tuong Vo, Ba-Ngu Vo,
    Antonio Cantoni
    """

    def __init__(self,
                 intensity_weights,
                 cardinality_probabilities,
                 cardinality_max,
                 kalman,
                 probability_survival,
                 probability_detection,
                 birth_cardinality_probabilites,
                 birth_weights,
                 birth_states,
                 birth_covariances,
                 clutter_intensity_function,
                 clutter_volume,
                 clutter_cardinality_function):

        # in paper represents the weights in mixture of state intensity
        # denoted by w and defined in equation (22)
        self._intensity_weights = intensity_weights
        # in paper defined as probability that given number of targets exist in frame
        # first occurs in equation (23)
        self._cardinality_probabilities = cardinality_probabilities
        # maximum number cardinality probability components
        self._cardinality_max = cardinality_max
        self._kalman: Kalman = kalman
        # in paper denoted as p_S,k defined in equation (19)
        self._probability_survival = probability_survival
        # in paper denoted as p_D,k defined in equation (20)
        self._probability_detection = probability_detection
        # probabilities that given number of targets are born at given instance
        # defined in paper in equation (11)
        self._birth_cardinality_probabilities = birth_cardinality_probabilites
        # in paper denoted as weights of gaussian components of birth intensity
        # occurs in equation (21)
        self._birth_weights = birth_weights
        # states of gaussian birth intensity as defined in equation (21)
        self._birth_states = birth_states
        # covariance of gaussian birth intensity as defined in equation (22)
        self._birth_covariances = birth_covariances
        self._clutter_intensity_functions = clutter_intensity_function
        self._clutter_volume = clutter_volume
        self._clutter_cardinality_function = clutter_cardinality_function

    def process(self, measurements):
        """
        Computes the prediction and update steps for CPHD filter for given set of input measurements

        The result is updated self._kalman._states, self._kalman._covariances, self._intensity_weights,
        self._cardinality_probabilities.

        The filtered states and covarinces are stored in self._kalman._states and in self._kalman._covariances.
        The number of targets with a given state is given by matching self._intensity_weights. The states are stored
        as an 2D array, where each row represents a single state. The covariances are stored in 3D matrix, where each
        matrix is a covariance for a given state. The intensity weights is a row vector indicating the number of targets
        present for each state and covariance. Thus for i-th predicted states the state is given by i-th row in states
        and i-th covariance matrix entry along first dimension, and i-th entry along the row of intensity weights
        indicates the number of targets targets with given state and covariance.

        The self._cardinality_probabilities is a row vector which indicates the probability that given number of targets
        is present.

        The input measurements should be a 2D matrix, where each row represents a single row.
        :param measurements: A 2D array of measurements
        """

        # computes the first summand of equation (24)
        predicted_intensity_weights, predicted_states, predicted_covariances = self._predict_intensity()
        # implements the equation (23)
        predicted_cardinality_probabilities = self._predict_cardinality_probabilities()

        # computes equations (41), (40), (39), (37), (36)
        self._kalman.state = predicted_states
        self._kalman.covariances = predicted_covariances
        self._kalman.update(measurements)

        # computes equation (35) and (34) for all measurements
        q_matrix = self._compute_q_matrix(measurements)

        # compute complete lambda sequence for use in eq. (31). The first row are values without knockouts.
        lambda_sequence, lambda_sequence_knockouts = self._compute_lambda_cap(predicted_intensity_weights,
                                                                              measurements,
                                                                              q_matrix)

        # compute the elementary symmetric functions of orders up to |Z|, and with all knockouts as use in equation (38)
        # The matrix is organized so that that first index is along the order of elementary function, while
        # the first entry along each row contains value without knockout, and value at ith index represents the value
        # with (i-1)th measurement knocked out.

        elementary_function_values, elementary_function_values_knockouts = self._compute_elementary_symmetric_functions(
            lambda_sequence,
            lambda_sequence_knockouts)

        # computes the gamma_k sequences necessary for other equations ( as they appear in eq. (30) )

        # here we compute the equation (31), where for each set of measurements the result is a row vector, whose
        # length is equal to length of predicted probabilities vector. Thus the shape of this matrix is (|Z|+1) x n
        # where n is number predicted probabilities and |Z| is number of measurements
        number_of_predictions = max(predicted_cardinality_probabilities.shape)
        zero_order_complete_gamma, _ = self._compute_gamma_k_sequence(predicted_intensity_weights,
                                                                      elementary_function_values,
                                                                      None,
                                                                      0,
                                                                      number_of_predictions)
        first_order_complete_gamma, first_order_knockout_gamma = self._compute_gamma_k_sequence(
            predicted_intensity_weights,
            elementary_function_values,
            elementary_function_values_knockouts,
            1,
            number_of_predictions)
        self._update_cardinality_distribution(zero_order_complete_gamma, predicted_cardinality_probabilities)
        self._update_intensity(zero_order_complete_gamma,
                               first_order_complete_gamma,
                               first_order_knockout_gamma,
                               q_matrix,
                               predicted_intensity_weights,
                               predicted_states,
                               predicted_covariances,
                               predicted_cardinality_probabilities,
                               measurements)

    def _predict_intensity(self):
        """
        Computes the predicted state intensity for given set of states and covariances. This method implements the
        equations (22),(23),(24),(25),(26),(27) from named paper.

        The number of predicted components is equal to number of input components plus the number of components included
        in birth variables. Thus for example if previous step ended with 10 components and birth variables include 5
        components then the result of this function is going to return 15 components.
        :return:
        """

        # Computes the weights used in equation (25)
        predicted_intensity_weights = self._predict_intensity_weights()
        # Computes the equations (26), (27)
        self._kalman.predict()
        predicted_intensity_weights.reshape((1, -1))
        states = self._kalman.state
        covariances = self._kalman.covariances
        # completes the computation of weights in equation (24) by attaching the birth weights
        predicted_weights = np.concatenate(predicted_intensity_weights, self._birth_weights, axis=1)
        # attach birth states to states
        predicted_states = np.concatenate(states, self._birth_states, axis=0)
        # attach covariances of birth states to covariances
        predicted_covariances = np.concatenate(covariances, self._birth_covariances)
        return predicted_weights, predicted_states, predicted_covariances

    def _predict_intensity_weights(self):
        # implements the equation (25) to compute weights, states and covariances
        predicted_intensity_weights = self._probability_survival * self._intensity_weights
        return predicted_intensity_weights

    def _predict_cardinality_probabilities(self):
        """
        Computes predicted probabilities of number of targets present as laid out in equation (23).

        This method returns the row vector of predicted target cardinality probabilities. The shape of the returned
        vector is 1x(n+m-1) where n is length of previous target cardinality probability vector, while m is the length
        of the target birth cardinality probability vector.
        :return:
        """
        # implements the equation (23) as convolution of inner sum and outer sum
        # Note that inner sum is only zero if l is larger then number of elements in previous vector of cardinality
        # probabilities
        inner_sum_upper_limit = max(self._cardinality_probabilities.shape)
        l_idx, j_idx = np.indices((inner_sum_upper_limit, inner_sum_upper_limit))
        # this function returns zero if j_idx is larger then j_idx, this thus ensures that zero entries are on
        # right positions, thus enabling us to simply perform summation over inner index
        first_factor = spec.binom(l_idx, j_idx)
        second_factor = self._cardinality_probabilities[0, l_idx]
        third_factor = self._probability_survival ** j_idx
        fourth_factor = (1 - self._probability_survival) ** (l_idx - j_idx)
        inner_summation_product = first_factor * second_factor * third_factor * fourth_factor
        inner_summation = np.squeeze(np.sum(inner_summation_product, axis=0))
        squezzed_birth = np.squeeze(self._birth_cardinality_probabilities)
        return np.convolve(squezzed_birth, inner_summation).reshape((1, -1))

    def _compute_gamma_k_sequence(self,
                                  weights,
                                  elementary_function_complete,
                                  elementary_function_knockouts,
                                  order,
                                  number_of_predictions):
        """
        This method computes the equation (31) for given order u and given set of elementary function values.

        If elementary_function_knockouts is set to null, we do not compute the values of the function for sets of
        measurements with knockouts and computation is skipped, to save resources.

        From equation (31) you can note that this is an infinite sequence, but from equations (29) and (30) we note
        that we need only first n values, where n is the number of components in predicted target cardinality
        probability vector.

        Two values are returned by this method.
        The first value is a row vector of length equal to length of predicted target cardinality probability vector.
        The second value is a matrix vector of shape kxm, where k is number of measurements in this cycle and m is
        number of columns in first return value.

        :param weights:
        :param elementary_function_complete:
        :param elementary_function_knockouts:
        :param order:
        :param number_of_predictions:
        :return:
        """

        num_measurements = elementary_function_complete.shape[1]
        idx_upper_limit = min(number_of_predictions, num_measurements + 1)
        n_idx, j_idx = np.indices((number_of_predictions, idx_upper_limit))

        first_factor = spec.factorial(num_measurements - j_idx)
        second_factor = self._clutter_cardinality_function(num_measurements - j_idx)
        third_factor = spec.factorial(n_idx) / spec.factorial(n_idx - j_idx - order)
        fourth_factor_upper = (1 - self._probability_detection) ** (n_idx - j_idx - order)
        fourth_factor_lower = np.sum(weights) ** (j_idx + order)
        fourth_factor = fourth_factor_upper / fourth_factor_lower
        fifth_factor = elementary_function_complete[0, 0:idx_upper_limit]
        product_without_knockouts = first_factor * second_factor * third_factor * fourth_factor * fifth_factor
        product_without_knockouts = np.tril(product_without_knockouts)
        result_without_knockouts = np.sum(product_without_knockouts, axis=1).reshape((1, -1))

        if elementary_function_knockouts is None:
            return result_without_knockouts, None
        else:
            num_measurements = num_measurements - 1
            idx_upper_limit = min(number_of_predictions, num_measurements + 1)
            n_idx, j_idx = np.indices((number_of_predictions, idx_upper_limit))

            first_factor = spec.factorial(num_measurements - j_idx)
            second_factor = self._clutter_cardinality_function(num_measurements - j_idx)
            third_factor = spec.factorial(n_idx) / spec.factorial(n_idx - j_idx - order)
            fourth_factor_upper = (1 - self._probability_detection) ** (n_idx - j_idx - order)
            fourth_factor_lower = np.sum(weights) ** (j_idx + order)
            fourth_factor = fourth_factor_upper / fourth_factor_lower
            fifth_factor = elementary_function_knockouts[:, np.newaxis, 0:idx_upper_limit]
            product_with_knockouts = first_factor * second_factor * third_factor * fourth_factor * fifth_factor
            product_with_knockouts = np.tril(product_with_knockouts)
            result_with_knockouts = np.sum(product_with_knockouts, axis=-1).squeeze()
            return result_without_knockouts, result_with_knockouts

    def _compute_lambda_cap(self, weights, measurements, q_matrix):
        """
        This method implements the equation (32) for sets of measurements. The sets for which it is computed is passed
        in set of measurements, and all those sets with one of measurements included.

        If the set of measurements is empty, then two empty vectors are returned.

        If the set of measurements contains n measurements, then two values are returned. First is the row vector
        representing the equation (32) for all measurements ( one value per measurement vector ). The second returned
        value is a 2D matrix, whose i-th row contains the computation of equation (32) with i-th measurement removed

        The shape of first return value is 1xn, where n is the number of measurements.
        The shape os second return value is nx(n-1), where n is the number of measurements in the set.

        :param weights: The variable w from the equation ( here as row vector )
        :param measurements: the variable Z from the equation
        :param q_matrix: the matrix of q-vectors, where each row is denoted by q_k(z) in the equation
        :return: A 2D array
        """
        # this method implements the equation (32)
        # The denominator of the first factor in equation
        clutter_values = self._clutter_intensity_functions(measurements)
        # the complete first factor in the equation
        clutter_factor = self._clutter_volume / clutter_values * self._probability_detection
        # compute the equation (32) for each measurement
        product = clutter_factor * weights @ q_matrix.T

        # in order to perform the computation for knockouts, it suffices to repeat the row and exclude diagonal elements
        num_meas = measurements.shape[0]
        repeated_product = np.repeat(product, num_meas, axis=0)
        knocked_out_array = repeated_product[~np.eye(num_meas, dtype=bool)].reshape((num_meas, -1))
        return product, knocked_out_array

    @staticmethod
    def _update_cardinality_distribution(zero_order_gamma, predicted_cardinality_probabilities):
        """
        Computes the updated target cardinality probability vector according to equation (29)
        :param zero_order_gamma:
        :param predicted_cardinality_probabilities:
        :return:
        """
        intermediate_product = zero_order_gamma * predicted_cardinality_probabilities
        inner_product = np.sum(intermediate_product, axis=1)
        updated_cardinality = intermediate_product / inner_product
        return updated_cardinality

    def _update_intensity(self,
                          zero_order_complete_gamma,
                          first_order_complete_gamma,
                          first_order_knockout_gamma,
                          q_matrix,
                          predicted_weights,
                          predicted_states,
                          predicted_covariances,
                          predicted_cardinality,
                          measurements):

        zero_complete_probability_inner_product = np.sum(zero_order_complete_gamma * predicted_cardinality, axis=1)
        first_complete_probability_inner_product = np.sum(first_order_complete_gamma * predicted_cardinality, axis=1)
        first_knockout_probability_inner_product = np.sum(first_order_knockout_gamma * predicted_cardinality,
                                                          axis=1).reshape((1, -1))

        miss_weights, miss_states, miss_covariances = self._compute_miss_intensity(
            zero_complete_probability_inner_product,
            first_complete_probability_inner_product,
            predicted_weights,
            predicted_states,
            predicted_covariances)
        detection_weights, detection_states, detection_covariances = self._compute_detect_intensity(
            zero_complete_probability_inner_product,
            first_knockout_probability_inner_product,
            q_matrix,
            predicted_weights,
            measurements)

        updated_weights = np.concatenate(miss_weights, detection_weights, axis=1)
        updated_states = np.concatenate(miss_states, detection_states, axis=0)
        updated_covariances = np.concatenate(miss_covariances, detection_covariances, axis=0)
        self._intensity_weights = updated_weights
        self._kalman.state = updated_states
        self._kalman.covariances = updated_covariances

    def _compute_miss_intensity(self, lower, upper, weights, states, covariances):
        result_weights = upper / lower * (1 - self._probability_detection) * weights
        return result_weights, states, covariances

    def _compute_detect_intensity(self, zero_ip, first_ip, q_matrix, weights, measurements):
        first_factor = self._probability_detection
        second_factor = weights
        third_factor = q_matrix
        fourth_factor = first_ip.T / zero_ip
        fifth_factor = self._clutter_volume / self._clutter_intensity_functions(measurements)
        result_weights = first_factor * second_factor * third_factor * fourth_factor * fifth_factor
        result_weights = result_weights.reshape((1, -1))
        result_states = self._kalman.state
        result_covariances = self._kalman.covariances
        return result_weights, result_states, result_covariances

    def _compute_q_matrix(self, measurements):
        """
        Computes the equation (34) for each measurement and each predicted state

        The return value of this computation is a 2D matrix, such that ij-entry is the q-factor from equation (35)
        for the i-th input measurement and the j-th predicted state.

        The shape of the returned array is nxm where n is number of measurements and m is number of predicted states

        :param measurements: a 2D array of measurements, where rows are treated as measurements
        :return: a 2D array of results
        """
        # computes equation (35)
        # creates a matrix such that i,j entry is equation (35) evaluated at i-th measurement with j as used in equation
        predicted_measurements = self._kalman.expected_measurements
        inv_covariance = self._kalman.inv_innovation_covariances
        dim = inv_covariance.shape[-1]
        # computes the difference between each measurement and each state and places them in three dimensional matrix
        # such that the i-th matrix contains the difference between the i-th measurement and all states
        diff = measurements[:, np.newaxis] - predicted_measurements

        # the i-th row in diff needs to be multiplied by i-th matrix, thus we need to expand the dimension of diff
        expanded_diff = np.expand_dims(diff, axis=-2)
        expanded_diff_transpose = np.transpose(expanded_diff, axes=(0, 1, -1, -2))
        exponent = np.squeeze((-1 / 2) * expanded_diff @ inv_covariance @ expanded_diff_transpose)

        # now exponent is a 2D matrix, and we need to compute a scaling factor for each row
        covariance = self._kalman.innovation_covariances
        determinants = np.linalg.det(covariance)[:, np.newaxis]
        scaling_inverse_squared = (2 * np.pi) ** dim * determinants
        scaling = 1 / np.sqrt(scaling_inverse_squared)

        q_matrix = scaling * np.exp(exponent)
        return q_matrix

    @staticmethod
    def _compute_elementary_symmetric_functions(lambda_sequence, lambda_sequence_knockouts):
        """
        Computes the elementary symmetric functions for sets of values contained in rows of inputs, of all orders.

        The rows of lambda_sequence and lambda_sequence_knockouts are sets of input values for which to compute the
        elementary symmetric functions of all orders. For example if rows of lambda_sequence have n entries, then
        elementary symmetric functions of orders 0,1,...,n are computed for each row of lambda_sequence. Same goes for
        second input.

        The method returns two values. The first is the value of elementary symmetric function of all orders for the
        row in lambda_sequence. The shape of the returned first value is 1x(n+1), where n is number of columns in
        lambda_sequence and number of measurements given to input process.

        The second returned value is the value of elementary symmetric function of all orders for each row in
        lambda_sequence_knockouts. The shape of the returned value ( it is a matrix of course ) is nxn, where n is
        the number of measurements given to the CPHD process, and n-1 is the number of columns in the
        lambda_sequence_knockouts.

        The computation of elementary symmetric functions is Newton-Girard formulae or Vietas theorem ( as outlined
        in the paper )

        :param lambda_sequence:
        :param lambda_sequence_knockouts:
        :return:
        """
        # polynomials which have lambda sequence entries as roots
        lambda_poly = np.poly(lambda_sequence)
        knockout_polys = [np.poly(lambda_sequence_knockouts[i]) for i in range(lambda_sequence_knockouts.shape[0])]
        knockout_polys = np.concatenate(knockout_polys, axis=0)

        dim_first = lambda_poly.shape[1]
        dim_second = knockout_polys.shape[1]
        first_pows = np.arange(dim_first)
        second_pows = np.arange(dim_second)

        first_result = (-1) ** first_pows * lambda_poly / lambda_poly[0]
        second_result = (-1) ** second_pows * knockout_polys / knockout_polys[0]

        return first_result, second_result
