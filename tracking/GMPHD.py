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
    transit_states = spawn_offsets[:, :, np.newaxis] + transit_states
    transit_states = np.transpose(transit_states, (2, 0, 1))
    new_states = np.reshape(transit_states, (new_weights.shape[1], -1))

    # compute covariance for each new state and weight
    transit_covs = spawn_covariance + spawn_transitions @ covariances[:, np.newaxis] @ np.transpose(spawn_transitions,
                                                                                                    (0, 2, 1))
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
    num_meas = measurements.shape[0]
    num_states = states.shape[0]
    mult_weights = np.repeat(weights, num_meas, 1)
    mult_states = np.repeat(states, num_meas, 0)
    mult_covs = np.repeat(covariances, num_meas, 0)
    mult_measurements = np.repeat(measurements, (num_states, 1))
    inno_cov, inv_inno_cov, kalman_gain = kf.compute_update_matrices(covariances,
                                                                     measurement_model,
                                                                     measurement_covariance)

    innovation = mult_measurements - mult_states @ measurement_model.T
    updated_states, updated_covariances = kf.pure_update(mult_states,
                                                         mult_covs,
                                                         innovation,
                                                         kalman_gain,
                                                         measurement_model)

    inno_dets = np.linalg.det(2 * np.pi * inno_cov)
    exponent = innovation[:, np.newaxis] @ inv_inno_cov @ np.transpose(innovation[:, np.newaxis], (0, 2, 1))
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
