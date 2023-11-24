from typing import Tuple

import numpy as np

from utils import mle_precision


class LinearGP:
    def __init__(self, noise_precision: float = None):
        self.noise_precision = noise_precision

    def _amplitude(self, X: np.ndarray):
        return np.eye(X.shape[1]) / 1e-5

    def kernel_fn(self, X: np.ndarray, Z: np.ndarray):
        return X @ self._amplitude(X) @ Z.T

    def _posterior_noise_precision(self, X: np.ndarray, y: np.ndarray):
        if self.noise_precision is None:
            return mle_precision(X, y)
        return self.noise_precision

    def _noise_free_posterior(
        self, query_mean: np.ndarray, X_obs: np.ndarray, y_obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        noise_variance = 1 / self._posterior_noise_precision(X_obs, y_obs)
        # Kernel of the observations
        kernel_train = self.kernel_fn(X_obs, X_obs)
        kernel_train += noise_variance * np.eye(len(X_obs))
        # Kernel of observations vs. query points
        kernel_train_test = self.kernel_fn(X_obs, query_mean)
        # Kernel of query points
        kernel_test = self.kernel_fn(query_mean, query_mean)
        kernel_test += noise_variance * np.eye(len(query_mean))
        # Compute posterior
        solved = (np.linalg.inv(kernel_train) @ kernel_train_test).T
        mean = solved @ y_obs
        covariance = kernel_test - solved @ kernel_train_test
        # For now we are only interested in the diagonal
        sdev = np.sqrt(covariance.diagonal().reshape(-1, 1))
        return mean, sdev

    def posterior(
        self,
        X_obs: np.ndarray,
        y_obs: np.ndarray,
        query_mean: np.ndarray,
        query_covariance: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Gaussian Process posterior moments with possibly noisy query points.

        Args:
            X_obs (np.ndarray): the array of index points corresponding to the observations.
            y_obs (np.ndarray): the array of (noisy) observations.
            query_mean (np.ndarray): the array of index point means at which the resulting posterior
                predictive distribution over function values is defined.
            query_covariance (np.ndarray): the corresponding array of index point covariances. The
                shape of this array will be (M, M, N), where M is the number of features and N is
                the number of query points.
        Returns:
            Tuple[np.ndarray, np.ndarray]: mean and standard deviation of Gaussian posterior
                predictive distribution.
        """
        noise_variance = 1 / self._posterior_noise_precision(X_obs, y_obs)
        mean, sdev = self._noise_free_posterior(query_mean, X_obs, y_obs)
        if query_covariance is None:
            # If no covariance is provided, the noise-free posterior is returned
            # to avoid unecessary computations (i.e., we would get the same result
            # assuming the covariance was a zero matrix)
            return mean, sdev
        # Kernel of the observations
        kernel_train = self.kernel_fn(X_obs, X_obs)
        kernel_train += noise_variance * np.eye(len(X_obs))
        kernel_train_inv = np.linalg.inv(kernel_train)
        # Extract the weights applied to the kernel of the query points when computing
        # the posterior predictive mean
        weights = kernel_train_inv @ y_obs
        # We can decompose the variance into that given by the noise-free posterior
        variance = np.square(sdev)
        # With additional correction terms
        variance += np.trace(self._amplitude(X_obs) @ query_covariance)
        variance -= np.trace(
            self._amplitude(X_obs)
            @ X_obs.T
            @ (kernel_train_inv - np.outer(weights, weights))
            @ X_obs
            @ self._amplitude(X_obs)
            @ query_covariance
        )

        return mean, np.sqrt(variance)
