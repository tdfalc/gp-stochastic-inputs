from typing import Tuple
from enum import Enum

import numpy as np

from utils import chain_combinations, ols_solution, mle_precision
from gp import LinearGP


class ImputationMethod(Enum):
    mean = "mean"
    ols = "ols"
    gp = "gp"


class Imputer:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.indices = np.arange(self.X.shape[1])
        self._noise_precisions = {}
        self._init_noise_precisions()

    def _init_noise_precisions(self):
        # As we use here only a linear kernel with additive noise, we can initialize
        # the observation noise variance using the maximum likelihood estimate
        # with the standard closed form solution. We initialize them simply
        # to avoid making the same calculation more than once.
        indices = np.arange(self.X.shape[1])
        for i in indices:
            for c in chain_combinations(set(indices) - {i}, 1, len(indices)):
                X, y = self.X[:, c], self.X[:, i]
                self._noise_precisions[(c, i)] = mle_precision(X, y)

    def _mean(self, i: int, *_):
        return np.mean(self.X[:, i]), 0

    def _ols(self, i: int, mean: np.ndarray, not_missing: np.ndarray):
        X, y = self.X[:, not_missing], self.X[:, i]
        return mean[:, not_missing] @ ols_solution(X, y), 0

    def _gp(self, i: int, mean: np.ndarray, not_missing: np.ndarray):
        noise_precision = self._noise_precisions[(tuple(not_missing), i)]
        gp = LinearGP(noise_precision)
        mean, sdev = gp.posterior(
            self.X[:, not_missing], self.X[:, i], mean[:, not_missing]
        )
        return mean, np.square(sdev)

    def impute(
        self,
        x: np.ndarray,
        missing_indicator: np.ndarray,
        method: ImputationMethod,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Impute missing values using specified method.

        Args:
            mean (np.ndarray): the index point.
            missing_indicator (np.ndarray): binary array containing a 1 if the feature value
                is missing, else a 0.
            method (ImputationMethod): method for which to perform imputation.
        Returns:
            Tuple[np.ndarray, np.ndarray]: mean and standard deviation of Gaussian posterior
        """
        if method not in ImputationMethod:
            raise ValueError(f"Unsupported imputation method: {method}")
        mean, covariance = x.copy(), np.zeros((x.shape[1], x.shape[1]))
        for i in self.indices[missing_indicator]:
            mean[:, i], covariance[i, i] = getattr(self, f"_{method.value}")(
                i, mean, self.indices[~missing_indicator]
            )
        return mean, covariance
