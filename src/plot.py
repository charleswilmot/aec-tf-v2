import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from scipy.stats import gaussian_kde
import numpy as np


class FigureManager:
    def __init__(self, filepath, save=True):
        self._fig = plt.figure(figsize=(8.53, 4.8), dpi=200)
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


def predicted_recerr_wrt_error(ax, errors, reconstruction_errors, actions, values, reward_scaling=600,
        ylim=[0, 0.04], title=None, xlabel=None, ylabel=None, legend=False):
    # print("\n###")
    # print(np.concatenate([errors, values], axis=-1))
    # print("###\n")
    ax.plot(errors, reconstruction_errors, 'b-', alpha=0.6, linewidth=1)
    ax.axvline(0, color="k", linestyle="--")
    ax.axvline(-1, color="k", linestyle="--", alpha=0.3)
    ax.axvline(1, color="k", linestyle="--", alpha=0.3)
    x = errors + actions
    y = reconstruction_errors - values / reward_scaling
    c = actions
    s = 16 * np.abs(actions).flatten() + 0.6
    ax.scatter(x, y, s=s, c=c, cmap='seismic', vmin=-1.125, vmax=1.125)
    # other cmaps: seismic, winter
    #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)
    if ylabel is None:
        ax.set_yticks([])
    if legend:
        ax.legend()


def action_wrt_error(ax, errors, actions, yscale, title=None, xlabel=None, ylabel=None, nbins=17, range=(-1.125, 1.125)):
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


def action_wrt_error_individual(ax, errors, actions, yscale, title=None, xlabel=None, ylabel=None):
    for action in actions.squeeze().T:
        ax.plot(errors, action, color='k', alpha=0.15)
    ax.plot(errors, np.mean(actions.squeeze(), axis=-1), color='r')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylabel is None:
        ax.set_yticks([])


def data_wrt_episode_mean_std(ax, data, std=True, ylim=[-5, 5], title=None, xlabel=None, ylabel=None):
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


def data_wrt_episode_std_quantile(ax, data, ylim=None, title=None, xlabel=None, ylabel=None):
    x = np.arange(data.shape[-1])
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    q10 = np.quantile(data, 0.10, axis=0)
    q25 = np.quantile(data, 0.25, axis=0)
    q75 = np.quantile(data, 0.75, axis=0)
    q90 = np.quantile(data, 0.90, axis=0)
    ax.fill_between(x, q10, q90, color='b', alpha=0.25)
    ax.fill_between(x, q25, q75, color='b', alpha=0.25)
    ax.fill_between(x, mean - std, mean + std, color='r', alpha=0.25)
    # for y in data:
    #     ax.plot(x, y, color='k', alpha=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)
    if ylim is None:
        ax.set_yticks([])


def critic_error_wrt_episode(ax, critic, recerr, title=None, xlabel=None, ylabel=None):
    critic = critic[..., :-1]
    true_critic = (recerr[..., :-1] - recerr[..., 1:]) * 600
    # print(recerr)
    # print(true_critic)
    x = np.arange(critic.shape[-1])
    a = 0
    b = 0
    for critic_one_error, true_critic_one_error in zip(critic, true_critic):
        total = 2
        sub_ax = inset_axes(ax, height="100%", width="100%", bbox_to_anchor=(0.05, a / len(critic) + 0.015, 1.00, 1 / len(critic)), bbox_transform=ax.transAxes)
        if b != 0:
            sub_ax.set_yticks([])
        sub_ax.set_xticks([])
        for critic_one_error_one_stimulus, true_critic_one_error_one_stimulus in zip(critic_one_error, true_critic_one_error):
            sub_ax.fill_between(x, critic_one_error_one_stimulus, true_critic_one_error_one_stimulus, color='b', alpha=0.5)
            sub_ax.plot(x, critic_one_error_one_stimulus, 'r-', alpha=1)
            total -= 1
            if total == 0:
                break
            b += 1
        a += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])


def scatter_plot(ax, data_x, data_y, title=None, xlabel=None, ylabel=None):
    epsilon = 1e-4
    mini_x = np.min(data_x)
    maxi_x = np.max(data_x)
    mini_y = np.min(data_y)
    maxi_y = np.max(data_y)
    if maxi_x - mini_x > epsilon and maxi_y - mini_y > epsilon:
        try:
            kde = gaussian_kde(np.vstack([data_x.flatten(), data_y.flatten()]))
            X, Y = np.mgrid[mini_x:maxi_x:100j, mini_y:maxi_y:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kde(positions).T, X.shape)
            ax.imshow(
                np.rot90(Z),
                cmap=plt.cm.gist_earth_r,
                extent=[mini_x - epsilon, maxi_x + epsilon, mini_y - epsilon, maxi_y + epsilon],
                aspect='auto'
            )
        except np.linalg.LinAlgError:
            pass
        try:
            a, b = np.polyfit(data_x.flatten(), data_y.flatten(), 1)
            ax.plot([mini_x, maxi_x], [a * mini_x + b, a * maxi_x + b], 'r-')
        except np.linalg.LinAlgError:
            pass
    ax.scatter(data_x.flatten(), data_y.flatten(), alpha=0.01, s=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


if __name__ == '__main__':
    from test_data import TestDataContainer
    import sys

    path = sys.argv[1]
    plot_path = "/".join(path.split("/")[:-2]) + "/plots/" + path.split("/")[-1].split(".")[0]
    print(plot_path)
    save = True

    test_data = TestDataContainer.load(path)
    test_data.plot(plot_path, save=save)
