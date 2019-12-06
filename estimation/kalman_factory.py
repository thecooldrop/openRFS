import numpy as np


def cv_model(timestep, acceleration_variance, num_vels):
    t = timestep
    kron_core = np.eye(num_vels)
    transition_core = np.reshape([1, t, 0, 1], (2, 2))
    transition_matrix = np.kron(kron_core, transition_core)

    noise_core = np.reshape([1 / 3 * t ** 3, 1 / 2 * t ** 2, 1 / 2 * t ** 2, t], (2, 2))
    noise_matrix = np.kron(kron_core, noise_core) * acceleration_variance

    return transition_matrix, noise_matrix


def ca_model(timestep, jerk_variance, num_vels):
    t = timestep
    kron_core = np.eye(num_vels)
    transition_core = np.reshape([1, t, 1 / 2 * t ** 2, 0, 1, t, 0, 0, 1], (3, 3))
    transition_matrix = np.kron(kron_core, transition_core)

    noise_core = np.reshape(
        [1 / 20 * t ** 5, 1 / 8 * t ** 4, 1 / 6 * t ** 3, 1 / 8 * t ** 4, 1 / 3 * t ** 3, 1 / 2 * t ** 2,
         1 / 6 * t ** 3, 1 / 2 * t ** 2, t],
        (3, 3))
    noise_matrix = np.kron(kron_core, noise_core) * jerk_variance
    return transition_matrix, noise_matrix
