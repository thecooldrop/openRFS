import numpy as np
import estimation.kf as kf


def process(weights,
            states,
            covariances,
            birth_weights,
            birth_states,
            birth_covariances,
            spawn_weights,
            spawn_offsets,
            spawn_covariances,
            spawn_transition,
            transition_model,
            transition_covariance,
            measurement_model,
            measurement_covariance,
            probability_detection,
            probability_survival,
            clutter_lambda,
            clutter_density,
            measurements):
    """

    Computes complete process of GMPHD filter as described in paper by Ba-Ngu Vo
    Parameters:
    -----------
    weights: np.ndarray
            Prior weights specified as a row vector
    states: np.ndarray
            Prior states specified as two dimensional np.ndarray, where each row represents a prior state
    covariance: np.ndarray
            Prior covariances specified as three dimensional np.ndarray, where each entry in first dimension is a
            covariance matrix for matching state
    birth_weights: np.ndarray
            Weights for birth components given as a row vector
    birth_states: np.ndarray
            Birth states specified as two dimensional np.ndarray, where each row represents a birth state
    birth_covariances: np.ndarray
            Covariances of birth states specified as three dimensional np.ndarray, where each entry in first dimension
            is a covariance matrix for matching birth state
    spawn_weights: np.ndarray
            Weights of spawn components specified as a row vector
    spawn_offsets: np.ndarray
            A two dimensional np.ndarray of offsets relative to state, where spawn components are born. Each row is
            considered an offset vector.
    spawn_covariances: np.ndarray
            Covariance matrices for spawn components specified as three dimensional np.ndarray.
    spawn_transition: np.ndarray
            A three dimensional np.ndarray of transition matrices. Each entry along first dimension is considered a
            transition matrix.
    transition_model: np.ndarray
            A transition model given as 2D transition matrix. Each state makes the same transition.
    transition_covariance: np.ndarray
            Transition covariance matrix given as a 2D matrix. Each state gets same transition covariance.
    measurement_model: np.ndarray
            A 2D matrix representing the linear measurement model.
    measurement_covariance: np.ndarray
            A 2D matrix representing the measurement noise covariance matrix. Each state is impinged with same noise
    probability_detection: float
            A numeric value between 0 and 1, used to specify a probability that a state is measured
    probability_survival: float
            A numeric value between 0 and 1, used to specify that target will continue to exist between two measurements
    clutter_lambda: float
            A positive numeric value specifying the number of expected false alarm measurements in current time-step
    clutter_density: float
            Inverse of the volume of the subset of measurement space observed by sensor, since the false alarm density
            is modeled as uniform distributed over this subset.
    measurements: np.ndarray
            Set of measurements used to update the filter. Each row represents a single measurement.


    """
    new_spawn_weights, new_spawn_states, new_spawn_covs = compute_spawn_components(weights,
                                                                                   states,
                                                                                   covariances,
                                                                                   spawn_weights,
                                                                                   spawn_offsets,
                                                                                   spawn_covariances,
                                                                                   spawn_transition)

    prediction_weights, prediction_states, prediction_covariances = predict_surviving_components(weights,
                                                                                                 states,
                                                                                                 covariances,
                                                                                                 transition_model,
                                                                                                 transition_covariance,
                                                                                                 probability_survival)

    prediction_weights = np.concatenate((prediction_weights, birth_weights, new_spawn_weights), axis=1)
    prediction_states = np.concatenate((prediction_states, birth_states, new_spawn_states), axis=0)
    prediction_covariances = np.concatenate((prediction_covariances, birth_covariances, new_spawn_covs), axis=0)

    miss_weights, miss_states, miss_covariances = compute_miss_components(prediction_weights,
                                                                          prediction_states,
                                                                          prediction_covariances,
                                                                          probability_detection)

    updated_weights, updated_states, updated_covariances = update_components(prediction_weights,
                                                                             prediction_states,
                                                                             prediction_covariances,
                                                                             measurement_model,
                                                                             measurement_covariance,
                                                                             measurements,
                                                                             clutter_lambda,
                                                                             clutter_density,
                                                                             probability_detection)

    full_weights = np.concatenate((updated_weights, miss_weights), axis=1)
    full_states = np.concatenate((updated_states, miss_states), axis=1)
    full_covariances = np.concatenate((updated_covariances, miss_covariances), axis=0)
    return full_weights, full_states, full_covariances


def compute_spawn_components(weights,
                             states,
                             covariances,
                             spawn_weights,
                             spawn_offsets,
                             spawn_covariance,
                             spawn_transitions):
    # multiply each old weight with each of spawn_weights
    new_weights = np.reshape(weights.T @ spawn_weights, (1, -1))

    # compute transition of each state with each of transition matrices
    transit_states = spawn_transitions @ states.T
    # for each transition matrix add the corresponding offset vector
    transit_states = spawn_offsets[:, :, np.newaxis] + transit_states
    # reorder the axes, so that the second axis becomes the fastest varying and first second fastest varying
    transit_states = np.transpose(transit_states, (2, 0, 1))
    # reshape the transit_states so that each row represents a state
    new_states = np.reshape(transit_states, (new_weights.shape[1], -1))

    # compute covariance for each new state and weight
    transit_covs = spawn_covariance + spawn_transitions @ covariances[:, np.newaxis] @ np.transpose(spawn_transitions,
                                                                                                    (0, 2, 1))
    # reshape the covaraince into old 3D shape
    new_covs = np.reshape(transit_covs, (-1, covariances.shape[1], covariances.shape[2]))

    return new_weights, new_states, new_covs


def predict_surviving_components(weights,
                                 states,
                                 covariances,
                                 transition_model,
                                 transition_covariance,
                                 probability_survival):

    new_weights = probability_survival * weights
    new_states, new_covariances = kf.predict(states, covariances, transition_model, transition_covariance)
    return new_weights, new_states, new_covariances


def compute_miss_components(weights,
                            states,
                            covariances,
                            probability_detection):
    new_weights = (1 - probability_detection) * weights
    return new_weights, states, covariances


def update_components(weights,
                      states,
                      covariances,
                      measurement_model,
                      measurement_covariance,
                      measurements,
                      probability_detection,
                      clutter_lambda,
                      clutter_density):

    # extract number of states and number of measurements
    num_meas = measurements.shape[0]
    num_states = states.shape[0]
    # repeat each weight,state and covariance ( element-wise ) as many times as there are measurements, since each
    # state needs to be updated with each of the measurements passed into function
    mult_weights = np.repeat(weights, num_meas, 1)
    mult_states = np.repeat(states, num_meas, 0)
    mult_covs = np.repeat(covariances, num_meas, 0)
    # repeat the measurements matrix ( block-wise ) as many times as there are states
    mult_measurements = np.repeat(measurements, (num_states, 1))
    # compute the matrices needed for updating for each of input states and repeat them
    inno_cov, inv_inno_cov, kalman_gain = kf.compute_update_matrices(covariances,
                                                                     measurement_model,
                                                                     measurement_covariance)
    mult_inno_cov = np.repeat(inno_cov, num_meas, 0)
    mult_inv_inno_cov = np.repeat(inv_inno_cov, num_meas, 0)
    mult_kalman_gain = np.repeat(kalman_gain, num_meas, 0)
    # compute difference between each measurement and each state
    mult_innovation = mult_measurements - mult_states @ measurement_model.T
    # update each of states with each of the measurements
    updated_states, updated_covariances = kf.pure_update(mult_states,
                                                         mult_covs,
                                                         mult_innovation,
                                                         mult_kalman_gain,
                                                         measurement_model)

    inno_dets = np.reshape(np.linalg.det(2 * np.pi * mult_inno_cov), (1, -1))
    exponent = mult_innovation[:, np.newaxis] @ mult_inv_inno_cov @ np.transpose(mult_innovation[:, np.newaxis], (0, 2, 1))
    exponent = np.reshape(exponent, (1, -1))
    gaussians = (inno_dets ^ (-1 / 2)) * np.exp((-1 / 2) * exponent)

    scaled_weights = probability_detection * mult_weights * gaussians
    l_weights = np.reshape(scaled_weights, (num_meas, num_states))
    sum_weights = np.sum(l_weights, axis=1)
    sum_weights = np.reshape(sum_weights, (-1, 1))
    coeffs = clutter_lambda * clutter_density + sum_weights
    l_up_weights = l_weights / coeffs
    updated_weights = np.reshape(l_up_weights, (1, -1))

    return updated_weights, updated_states, updated_covariances
