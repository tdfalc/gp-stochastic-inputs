from typing import Sequence, List, Tuple
from itertools import combinations, chain
import joblib
import contextlib

import numpy as np


def chain_combinations(sequence: Sequence[int], min: int, max: int) -> List[Tuple]:
    return list(
        chain.from_iterable((combinations(sequence, i) for i in range(min, max + 1)))
    )


def ols_solution(X: np.ndarray, y: np.ndarray):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def mle_precision(X: np.ndarray, y: np.ndarray):
    return 1 / np.square(X @ ols_solution(X, y) - y).mean()


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
