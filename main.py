from pathlib import Path
from typing import Optional
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import stats

from gp_stochastic_inputs.gp import LinearGP


def plot_prediction(ax: Axes, mean: float, sdev: float, label: Optional[str] = None):
    xs = np.linspace(mean - 3 * sdev, mean + 3 * sdev, 100)
    ax.plot(xs, stats.norm.pdf(xs, mean, sdev), label=label)

    if label is not None:
        leg = ax.legend(loc="upper right")
        leg.get_frame().set_linewidth(0)


if __name__ == "__main__":

    np.random.seed(100)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12
    plt.rcParams["mathtext.fontset"] = "cm"  # Use CM for math font.

    savedir = Path(__file__).parent / "docs/"
    os.makedirs(savedir, exist_ok=True)

    gp = LinearGP()
    train_size, test_size = 50, 3
    coefficients = np.array([[0.5], [0.9]])

    # Generate observations
    sample_size = train_size + test_size
    X = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=sample_size)
    y = X @ coefficients + np.random.normal(0, 0.3, (sample_size, 1))

    # Train/test split
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[-test_size:], y[-test_size:]

    fig, axs = plt.subplots(test_size, 1, dpi=300, sharex=True, sharey=True)

    for i, (ax, x) in enumerate(zip(axs, X_test)):
        x = np.atleast_2d(x)
        mean, sdev = gp.posterior(X_train, y_train, query_mean=x)
        label = "Deterministic Inputs" if i == 0 else None
        plot_prediction(ax, mean.item(), sdev.item(), label)

        variance = np.random.uniform(0.01, 0.3)
        query_covariance = np.array([[variance, 0], [0, variance]])
        mean, sdev = gp.posterior(X_train, y_train, query_mean=x, query_covariance=query_covariance)
        label = "Stochastic Inputs" if i == 0 else None
        plot_prediction(ax, mean.item(), sdev.item(), label)

        ax.set_xlabel("Prediction")
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(savedir / "preds.png")
