import numpy as np
import random


class NoiseGenerator(object):

    def corrupt_input(self, noise_type, data, fraction_noise, seed_value):
        """
        Corrupt matrix or vector with masking and salt-and-pepper noise
        :param noise_type: masking or salt-and-pepper
        :param data:
        :param fraction_noise:
        :param seed_value:
        :return:
        """
        random.seed(seed_value)

        data_copy = data.copy()

        if noise_type == 'masking':
            x_corrupted = self._masking_noise_reload(data_copy, fraction_noise)
            # x_corrupted = self._masking_noise(data_copy, fraction_noise)
        elif noise_type == 'salt_and_pepper':
            min_value, max_value = 0, 1 # for binary data
            # x_corrupted = self._salt_and_pepper_noise(data_copy, fraction_noise, min_value, max_value)
            x_corrupted = self._salt_and_pepper_noise_reload(data_copy, fraction_noise)
        else:
            x_corrupted = data_copy
        return x_corrupted

    def _salt_and_pepper_noise_reload(self, X, v):
        """
        Apply salt and pepper noise to data in X, in other words a fraction v of elements of X
        (chosen at random) is set to its maximum or minimum value according to a fair coin flip.
        :param X: array_like, Input data
        :param v: int, fraction of elements to distort
        :return: transformed data
        """

        gb = X.copy()
        prob = v / 100.0

        rnd = np.random.rand(gb.shape[0], gb.shape[1])
        noisy = gb.copy()
        noisy[rnd < prob] = 0
        noisy[rnd > 1 - prob] = 1
        return noisy

    def _masking_noise_reload(self, X, v):
        x_noise = X.copy()
        v = v / 100.0

        x_noise[np.random.rand(*x_noise.shape) < v] *= 0

        return x_noise


    def _masking_noise(self, X, v):
        """
        Apply masking noise to data in X, in other words a fraction v of elements of X
        (chosen at random) is forced to zero.
        :param X: array_like, Input data
        :param v: int, fraction of elements to distort between [0, 100]
        :return: transformed data
        """
        X_noise = X.copy()
        X_noise_result = X_noise.copy()

        n_samples = X.shape[0]
        n_features = X.shape[1]

        v_percentage = round(n_features * v / 100)
        print('v_percentage', v_percentage)

        for i in range(n_samples):
            perm = np.random.permutation(n_features)
            utils = perm[0:v_percentage]
            x_i = X_noise[i][:]
            x_i[utils] = 0
            X_noise_result[i] = x_i

        return X_noise_result

    def _salt_and_pepper_noise(self, X, v, min_value, max_value):
        """
        Apply salt and pepper noise to data in X, in other words a fraction v of elements of X
        (chosen at random) is set to its maximum or minimum value according to a fair coin flip.
        :param X: array_like, Input data
        :param v: int, fraction of elements to distort between [0, 100]
        :return: transformed data
        """
        X_noise = X.copy()
        n_features = X.shape[1]
        v_percentage = round(n_features * v / 100)

        print('percentage: ', v_percentage)

        for i, sample in enumerate(X):
            perm = np.random.permutation(n_features)
            utils = perm[0:v_percentage]

            for m in utils:
                if np.random.random() < 0.5:
                    X_noise[i][m] = min_value
                else:
                    X_noise[i][m] = max_value

        return X_noise
