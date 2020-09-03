import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np


class FigureManager:
    def __init__(self, filepath, save=True):
        self._fig = plt.figure(dpi=200)
        self._filepath = filepath
        self._save = save

    def __enter__(self):
        return self._fig

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._save:
            print("saving plot {}  ...  ".format(self._filepath), end="")
            self._fig.savefig(self._filepath)
            print("done")
            plt.close(self._fig)
        else:
            plt.show()


def recerr_wrt_error(ax, errors, reconstruction_errors,
        ylim=[0, 0.04], title=None, xlabel=None, ylabel=None, inset=True, legend=False):
    for e, r in zip(errors, reconstruction_errors):
        ax.plot(e, r, 'b-', alpha=0.6, linewidth=1)
    mean = np.mean(reconstruction_errors, axis=0)
    ax.plot(e, mean, 'r-', linewidth=3, label="mean")
    ax.axvline(0, color="k", linestyle="--")
    if inset:
        axins = inset_axes(ax, width="20%", height="20%", borderpad=2)
        axins.plot(e, mean, 'r-')
        axins.axvline(0, color="k", linestyle="--", alpha=0.5)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("Mean only")
    #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)
    if ylabel is None:
        ax.set_yticks([])
    if legend:
        ax.legend()


def action_wrt_error(ax, errors, actions, yscale, title=None, xlabel=None, ylabel=None, nbins=100, range=(-1, 1)):
    histograms = []
    edges = []
    for err, acts in zip(errors, actions):
        hist, edg = np.histogram(acts, bins=nbins, density=True, range=range)
        histograms.append(hist)
        edges.append(edg)
    image = np.zeros((len(hist), len(histograms)))
    for i, hist in enumerate(histograms):
        image[:, i] = hist
    ax.imshow(image, origin="lower", extent=(errors[0], errors[-1], -yscale, yscale), aspect="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylabel is None:
        ax.set_yticks([])


def data_wrt_episode(ax, data, std=True, ylim=[-5, 5], title=None, xlabel=None, ylabel=None):
    mean = np.mean(data, axis=0)
    x = np.arange(len(mean))
    ax.plot(x, mean, 'b-')
    if std:
        std = np.std(data, axis=0)
        ax.fill_between(x, mean - std, mean + std, color='b', alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylabel is None:
        ax.set_yticks([])


if __name__ == '__main__':
    from test_data import TestDataContainer

    path = "../experiments/2020-09-03/14-22-11_experiment.policy_after.2000__experiment.test_at_start.True/tests/default_at_2m.pkl_000000.pkl"
    plot_path = "/tmp/plot/"
    save = True

    test_data = TestDataContainer.load(path)
    test_data.plot(plot_path, save=save)
