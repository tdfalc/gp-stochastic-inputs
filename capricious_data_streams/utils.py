from typing import Sequence, List, Tuple
from itertools import combinations, chain

import numpy as np


def chain_combinations(sequence: Sequence[int], min: int, max: int) -> List[Tuple]:
    return list(
        chain.from_iterable((combinations(sequence, i) for i in range(min, max + 1)))
    )


def ols_solution(X: np.ndarray, y: np.ndarray):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def mle_precision(X: np.ndarray, y: np.ndarray):
    return 1 / np.square(X @ ols_solution(X, y) - y).mean()
