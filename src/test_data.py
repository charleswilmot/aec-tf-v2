import os
import numpy as np
import pickle
import plot
from itertools import product, starmap
from collections import defaultdict


DEG_TO_PX = 320 / 90


dttest_case = np.dtype([
    ("stimulus", np.int32),
    ("object_distance", np.float32),
    ("vergence_error", np.float32),
    ("cyclo_pos", np.float32),
    ("pan_error", np.float32),
    ("tilt_error", np.float32),
    ("n_iterations", np.int32)
])
dttest_result = np.dtype([
    ("vergence_error", np.float32),
    ("pan_error", np.float32),
    ("tilt_error", np.float32),
    ("recerr_magno", np.float32),
    ("recerr_pavro", np.float32),
    ("tilt_return_estimates", np.float32),
    ("pan_return_estimates", np.float32),
    ("vergence_return_estimates", np.float32),
    ("cyclo_return_estimates", np.float32),
    ("pan_pos", np.float32),
    ("tilt_pos", np.float32),
    ("vergence_pos", np.float32),
    ("cyclo_pos", np.float32),
    ("pan_speed", np.float32),
    ("tilt_speed", np.float32),
    ("vergence_speed", np.float32),
    ("cyclo_speed", np.float32),
    ("pan_action", np.float32),
    ("tilt_action", np.float32),
    ("vergence_action", np.float32),
    ("cyclo_action", np.float32),
])

def dttest_data(n_iterations):
    return np.dtype([
        ("conf", dttest_case),
        ("result", (dttest_result, (n_iterations,))),
    ])

one_pixel = 90 / 320
min_action = one_pixel / 2


class TestDataContainer:
    def __init__(self, stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error, n_iterations, name="no_name"):
        self.default_stimulus = stimulus
        self.default_object_distance = object_distance
        self.default_vergence_error = vergence_error
        self.default_cyclo_pos = cyclo_pos
        self.default_pan_error = pan_error
        self.default_tilt_error = tilt_error
        self.default_n_iterations = n_iterations
        self.name = name
        self.data = {"test_description": {}}

    def dump(self, path, name=None):
        with open(path + "/" + "{}.pkl".format(self.name if name is None else name), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return sum([len(val) for key, val in self.data.items() if type(key) is int])

    @classmethod
    def load_test_description(cls, path):
        return cls.load(path).data["test_description"]

    def add(self, plot_name, **anchors):
        self.data["test_description"][plot_name] = anchors
        if anchors["n_iterations"] not in self.data:
            self.data[anchors["n_iterations"]] = np.zeros(0, dtype=dttest_data(anchors["n_iterations"]))
        self.data[anchors["n_iterations"]] = np.union1d(self.data[anchors["n_iterations"]], test_data_between(**anchors))

    def add_vergence_trajectory(self, stimulus=None, object_distance=None, vergence_error=None, n_iterations=None):
        IMPOSED_PAN_SPEED_ERROR = [0]
        IMPOSED_TILT_SPEED_ERROR = [0]
        IMPOSED_CYCLO_POS = [0]
        self.add("vergence_trajectory",
            stimulus=stimulus if stimulus is not None else self.default_stimulus,
            object_distance=object_distance if object_distance is not None else self.default_object_distance,
            vergence_error=vergence_error if vergence_error is not None else self.default_vergence_error,
            cyclo_pos=IMPOSED_CYCLO_POS,
            pan_error=IMPOSED_PAN_SPEED_ERROR,
            tilt_error=IMPOSED_TILT_SPEED_ERROR,
            n_iterations=n_iterations if n_iterations is not None else self.default_n_iterations
        )

    def add_cyclo_trajectory(self, stimulus=None, object_distance=None, cyclo_pos=None, n_iterations=None):
        IMPOSED_PAN_SPEED_ERROR = [0]
        IMPOSED_TILT_SPEED_ERROR = [0]
        IMPOSED_VERGENCE_ERROR = [0]
        self.add("cyclo_trajectory",
            stimulus=stimulus if stimulus is not None else self.default_stimulus,
            object_distance=object_distance if object_distance is not None else self.default_object_distance,
            vergence_error=IMPOSED_VERGENCE_ERROR,
            cyclo_pos=cyclo_pos if cyclo_pos is not None else self.default_cyclo_pos,
            pan_error=IMPOSED_PAN_SPEED_ERROR,
            tilt_error=IMPOSED_TILT_SPEED_ERROR,
            n_iterations=n_iterations if n_iterations is not None else self.default_n_iterations
        )

    def add_speed_trajectory(self, pan_or_tilt, stimulus=None, object_distance=None, speed_errors=None, n_iterations=None):
        IMPOSED_VERGENCE_ERROR = [0]
        IMPOSED_CYCLO_POS = [0]
        if pan_or_tilt == "pan":
            pan_error = speed_errors if speed_errors is not None else self.default_pan_error
            tilt_error = [0]
        else:
            pan_error = [0]
            tilt_error = speed_errors if speed_errors is not None else self.default_tilt_error
        self.add("{}_speed_trajectory".format(pan_or_tilt),
            stimulus=stimulus if stimulus is not None else self.default_stimulus,
            object_distance=object_distance if object_distance is not None else self.default_object_distance,
            vergence_error=IMPOSED_VERGENCE_ERROR,
            cyclo_pos=IMPOSED_CYCLO_POS,
            pan_error=pan_error,
            tilt_error=tilt_error,
            n_iterations=n_iterations if n_iterations is not None else self.default_n_iterations
        )

    def add_wrt_vergence_error(self, vergence_error_bound_in_px, stimulus=None, object_distance=None):
        IMPOSED_N_ITERATIONS = 1
        IMPOSED_PAN_SPEED_ERROR = [0]
        IMPOSED_TILT_SPEED_ERROR = [0]
        IMPOSED_CYCLO_POS = [0]
        vergence_error = vergence_error_bound_in_px * 90 / 320
        vergence_error = np.arange(-vergence_error, vergence_error + min_action, min_action)
        self.add("wrt_vergence_error",
            stimulus=stimulus if stimulus is not None else self.default_stimulus,
            object_distance=object_distance if object_distance is not None else self.default_object_distance,
            vergence_error=vergence_error,
            cyclo_pos=IMPOSED_CYCLO_POS,
            pan_error=IMPOSED_PAN_SPEED_ERROR,
            tilt_error=IMPOSED_TILT_SPEED_ERROR,
            n_iterations=IMPOSED_N_ITERATIONS
        )

    def add_wrt_cyclo_pos(self, cyclo_pos_bound_in_deg, n=50, stimulus=None, object_distance=None):
        IMPOSED_N_ITERATIONS = 1
        IMPOSED_PAN_SPEED_ERROR = [0]
        IMPOSED_TILT_SPEED_ERROR = [0]
        IMPOSED_VERGENCE_ERROR = [0]
        cyclo_pos = np.linspace(-cyclo_pos_bound_in_deg, cyclo_pos_bound_in_deg, n).astype(np.float32)
        self.add("wrt_cyclo_pos",
            stimulus=stimulus if stimulus is not None else self.default_stimulus,
            object_distance=object_distance if object_distance is not None else self.default_object_distance,
            vergence_error=IMPOSED_VERGENCE_ERROR,
            cyclo_pos=cyclo_pos,
            pan_error=IMPOSED_PAN_SPEED_ERROR,
            tilt_error=IMPOSED_TILT_SPEED_ERROR,
            n_iterations=IMPOSED_N_ITERATIONS
        )

    def add_wrt_speed_error(self, pan_or_tilt, speed_error_bound_in_px, stimulus=None, object_distance=None):
        IMPOSED_N_ITERATIONS = 1
        IMPOSED_VERGENCE_ERROR = [0]
        IMPOSED_CYCLO_POS = [0]
        speed_error = speed_error_bound_in_px * 90 / 320
        speed_errors = np.arange(-speed_error, speed_error + min_action, min_action)
        if pan_or_tilt == "pan":
            pan_error = speed_errors if speed_errors is not None else self.default_pan_error
            tilt_error = [0]
        else:
            pan_error = [0]
            tilt_error = speed_errors if speed_errors is not None else self.default_tilt_error
        self.add("wrt_{}_error".format(pan_or_tilt),
            stimulus=stimulus if stimulus is not None else self.default_stimulus,
            object_distance=object_distance if object_distance is not None else self.default_object_distance,
            vergence_error=IMPOSED_VERGENCE_ERROR,
            cyclo_pos=IMPOSED_CYCLO_POS,
            pan_error=pan_error,
            tilt_error=tilt_error,
            n_iterations=IMPOSED_N_ITERATIONS
        )

    def tests_by_chunks(self, n_iterations, chunk_size):
        if n_iterations not in self.data:
            raise ValueError("No test has a length of {}".format(n_iterations))
        tests = self.data[n_iterations]
        for i in range(0, len(tests), chunk_size):
            yield tests[i:i+chunk_size]

    def get_tests_lengths(self):
        return [x for x in self.data if type(x) is int]

    def data_by_name(self, name, dim0=None, sort_order=None):
        anchors = self.data["test_description"][name]
        data = self.data[anchors["n_iterations"]]
        where = np.logical_and.reduce([np.in1d(data["conf"][name], val) for name, val in anchors.items()])
        data = data[where]
        if dim0 is not None:
            first, second = dim0.split(".")
            unique, count = np.unique(data[first][second], return_counts=True)
            if (count != count[0]).any():
                raise ValueError("Impossible to factorize by {}: number of value is not balances. Got {}".format(dim0, dict(zip(unique, count))))
            data = data[np.argsort(data[first][second])]
            data = data.reshape((-1, count[0]))
            # print(data.shape, data[first][second].shape)
        if sort_order is not None:
            first, second = sort_order.split(".")
            arg = np.argsort(data[first][second], axis=1).squeeze()
            # print(data.shape, data[first][second].shape, arg.shape)
            data = np.take_along_axis(data, arg, axis=1)
        return data

    def missing_data(self, *args):
        for arg in args:
            if arg not in self.data["test_description"]:
                return True
        return False

    def plot_recerr_wrt_error(self, path, ylim=[0, 0.04], save=True):
        if self.missing_data("wrt_pan_error", "wrt_tilt_error", "wrt_vergence_error", "wrt_cyclo_pos"):
            return
        with plot.FigureManager(path + "/reconstruction_error.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("wrt_pan_error", dim0="conf.stimulus", sort_order="conf.pan_error")
            plot.recerr_wrt_error(
                ax,
                data["result"]["pan_error"] * DEG_TO_PX,
                data["result"]["recerr_magno"],
                ylim=ylim,
                xlabel="pan error (px/it)",
                ylabel="Reconstruction error",
            )

            ax = fig.add_subplot(142)
            data = self.data_by_name("wrt_tilt_error", dim0="conf.stimulus", sort_order="conf.tilt_error")
            plot.recerr_wrt_error(
                ax,
                data["result"]["tilt_error"] * DEG_TO_PX,
                data["result"]["recerr_magno"],
                ylim=ylim,
                xlabel="tilt error (px/it)",
            )

            ax = fig.add_subplot(143)
            data = self.data_by_name("wrt_vergence_error", dim0="conf.stimulus", sort_order="conf.vergence_error")
            plot.recerr_wrt_error(
                ax,
                data["result"]["vergence_error"] * DEG_TO_PX,
                data["result"]["recerr_pavro"],
                ylim=ylim,
                xlabel="vergence error (px)",
            )

            ax = fig.add_subplot(144)
            data = self.data_by_name("wrt_cyclo_pos", dim0="conf.stimulus", sort_order="conf.cyclo_pos")
            plot.recerr_wrt_error(
                ax,
                data["result"]["cyclo_pos"],
                data["result"]["recerr_pavro"],
                ylim=ylim,
                xlabel="cyclo error (deg)",
                legend=True
            )

    def plot_predicted_recerr_wrt_error(self, path, ylim=[0, 0.04], save=True):
        if self.missing_data("wrt_pan_error", "wrt_tilt_error", "wrt_vergence_error", "wrt_cyclo_pos"):
            return
        for stimulus in self.data["test_description"]["wrt_pan_error"]["stimulus"]:
            with plot.FigureManager(path + "/reconstruction_error_predicted_s{}.png".format(stimulus), save=save) as fig:
                ax = fig.add_subplot(141)
                data = self.data_by_name("wrt_pan_error", dim0="conf.stimulus", sort_order="conf.pan_error")
                plot.predicted_recerr_wrt_error(
                    ax,
                    data["result"]["pan_error"][stimulus] * DEG_TO_PX,
                    data["result"]["recerr_magno"][stimulus],
                    data["result"]["pan_action"][stimulus] * DEG_TO_PX,
                    data["result"]["pan_return_estimates"][stimulus],
                    ylim=ylim,
                    xlabel="pan error (px/it)",
                    ylabel="Reconstruction error",
                )

                ax = fig.add_subplot(142)
                data = self.data_by_name("wrt_tilt_error", dim0="conf.stimulus", sort_order="conf.tilt_error")
                plot.predicted_recerr_wrt_error(
                    ax,
                    data["result"]["tilt_error"][stimulus] * DEG_TO_PX,
                    data["result"]["recerr_magno"][stimulus],
                    data["result"]["tilt_action"][stimulus] * DEG_TO_PX,
                    data["result"]["tilt_return_estimates"][stimulus],
                    ylim=ylim,
                    xlabel="tilt error (px/it)",
                )

                ax = fig.add_subplot(143)
                data = self.data_by_name("wrt_vergence_error", dim0="conf.stimulus", sort_order="conf.vergence_error")
                plot.predicted_recerr_wrt_error(
                    ax,
                    data["result"]["vergence_error"][stimulus] * DEG_TO_PX,
                    data["result"]["recerr_pavro"][stimulus],
                    data["result"]["vergence_action"][stimulus] * DEG_TO_PX,
                    data["result"]["vergence_return_estimates"][stimulus],
                    ylim=ylim,
                    xlabel="vergence error (px)",
                )

                ax = fig.add_subplot(144)
                data = self.data_by_name("wrt_cyclo_pos", dim0="conf.stimulus", sort_order="conf.cyclo_pos")
                plot.predicted_recerr_wrt_error(
                    ax,
                    data["result"]["cyclo_pos"][stimulus],
                    data["result"]["recerr_pavro"][stimulus],
                    data["result"]["cyclo_action"][stimulus],
                    data["result"]["cyclo_return_estimates"][stimulus],
                    ylim=ylim,
                    xlabel="cyclo error (deg)",
                )

    def plot_action_wrt_error(self, path, save=True):
        if self.missing_data("wrt_pan_error", "wrt_tilt_error", "wrt_vergence_error", "wrt_cyclo_pos"):
            return
        with plot.FigureManager(path + "/policy.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("wrt_pan_error", dim0="conf.pan_error")

            plot.action_wrt_error(
                ax,
                data["conf"]["pan_error"][:, 0] * DEG_TO_PX,
                data["result"]["pan_action"] * DEG_TO_PX,
                np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 999], dtype=np.float32),
                xlabel="pan error (px/it)",
                ylabel="action\n(in px/it² for pan and tilt, in px/it for vergence, in deg/it for cyclo)"
            )

            ax = fig.add_subplot(142)
            data = self.data_by_name("wrt_tilt_error", dim0="conf.tilt_error")
            plot.action_wrt_error(
                ax,
                data["conf"]["tilt_error"][:, 0] * DEG_TO_PX,
                data["result"]["tilt_action"] * DEG_TO_PX,
                np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 999], dtype=np.float32),
                xlabel="tilt error (px/it)",
            )

            ax = fig.add_subplot(143)
            data = self.data_by_name("wrt_vergence_error", dim0="conf.vergence_error")
            plot.action_wrt_error(
                ax,
                data["conf"]["vergence_error"][:, 0] * DEG_TO_PX,
                data["result"]["vergence_action"] * DEG_TO_PX,
                np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 999], dtype=np.float32),
                xlabel="vergence error (px)",
            )

            ax = fig.add_subplot(144)
            ax.yaxis.tick_right()
            data = self.data_by_name("wrt_cyclo_pos", dim0="conf.cyclo_pos")
            plot.action_wrt_error(
                ax,
                data["conf"]["cyclo_pos"][:, 0],
                -data["result"]["cyclo_action"], # * DEG_TO_PX,
                np.array([-3.58, -1.79, 0.0, 1.79, 3.58, 999], dtype=np.float32),
                # [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 999],
                xlabel="cyclo pos (deg)",
                ylabel='',
            )

    def plot_action_wrt_error_individual(self, path, save=True):
        if self.missing_data("wrt_pan_error", "wrt_tilt_error", "wrt_vergence_error", "wrt_cyclo_pos"):
            return
        with plot.FigureManager(path + "/policy_individual.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("wrt_pan_error", dim0="conf.pan_error")
            plot.action_wrt_error_individual(
                ax,
                data["conf"]["pan_error"][:, 0] * DEG_TO_PX,
                data["result"]["pan_action"] * DEG_TO_PX,
                xlabel="pan error (px/it)",
                ylabel="action\n(in px/it² for pan and tilt, in px/it for vergence, in deg/it for cyclo)"
            )

            ax = fig.add_subplot(142)
            data = self.data_by_name("wrt_tilt_error", dim0="conf.tilt_error")
            plot.action_wrt_error_individual(
                ax,
                data["conf"]["tilt_error"][:, 0] * DEG_TO_PX,
                data["result"]["tilt_action"] * DEG_TO_PX,
                xlabel="tilt error (px/it)",
            )

            ax = fig.add_subplot(143)
            data = self.data_by_name("wrt_vergence_error", dim0="conf.vergence_error")
            plot.action_wrt_error_individual(
                ax,
                data["conf"]["vergence_error"][:, 0] * DEG_TO_PX,
                data["result"]["vergence_action"] * DEG_TO_PX,
                xlabel="vergence error (px)",
            )

            ax = fig.add_subplot(144)
            data = self.data_by_name("wrt_cyclo_pos", dim0="conf.cyclo_pos")
            plot.action_wrt_error_individual(
                ax,
                data["conf"]["cyclo_pos"][:, 0],
                -data["result"]["cyclo_action"],
                xlabel="cyclo pos (deg)",
            )

    def plot_abs_error_in_episode(self, path, save=True):
        if self.missing_data("pan_speed_trajectory", "tilt_speed_trajectory", "vergence_trajectory", "cyclo_trajectory"):
            return
        with plot.FigureManager(path + "/abs_error.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("pan_speed_trajectory")
            plot.data_wrt_episode_mean_std(
                ax,
                np.abs(data["result"]["pan_error"]) * DEG_TO_PX,
                std=True,
                ylim=[-0.1, 3.5],
                xlabel="#Iteration (pan)",
                ylabel="Joint error\n(px/it for pan and tilt, px for vergence, deg for cyclo)"
            )
            ax.axhline(1, color="k", linestyle="--", alpha=0.5)

            ax = fig.add_subplot(142)
            data = self.data_by_name("tilt_speed_trajectory")
            plot.data_wrt_episode_mean_std(
                ax,
                np.abs(data["result"]["tilt_error"]) * DEG_TO_PX,
                std=True,
                ylim=[-0.1, 3.5],
                xlabel="#Iteration (tilt)",
            )
            ax.axhline(1, color="k", linestyle="--", alpha=0.5)

            ax = fig.add_subplot(143)
            data = self.data_by_name("vergence_trajectory")
            plot.data_wrt_episode_mean_std(
                ax,
                np.abs(data["result"]["vergence_error"]) * DEG_TO_PX,
                std=True,
                ylim=[-0.1, 3.5],
                xlabel="#Iteration (vergence)",
            )
            ax.axhline(1, color="k", linestyle="--", alpha=0.5)

            ax = fig.add_subplot(144)
            data = self.data_by_name("cyclo_trajectory")
            plot.data_wrt_episode_mean_std(
                ax,
                np.abs(data["result"]["cyclo_pos"]),
                std=True,
                ylim=[-0.1, 3.5],
                xlabel="#Iteration (cyclo)",
            )

    def plot_error_in_episode(self, path, save=True):
        if self.missing_data("pan_speed_trajectory", "tilt_speed_trajectory", "vergence_trajectory", "cyclo_trajectory"):
            return
        with plot.FigureManager(path + "/error_trajectory.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("pan_speed_trajectory")
            errors = data["conf"]["pan_error"]
            ylim = [-4, 4]
            plot.data_wrt_episode_std_quantile(
                ax,
                data["result"]["pan_error"] * DEG_TO_PX,
                ylim=ylim,
                xlabel="#Iteration (pan)",
                ylabel="Joint error\n(px/it for pan and tilt, px for vergence, deg for cyclo)"
            )
            ax.axhline(1, color="k", linestyle="--", alpha=0.5)
            ax.axhline(-1, color="k", linestyle="--", alpha=0.5)
            ax.legend(prop={'size': 5})

            ax = fig.add_subplot(142)
            data = self.data_by_name("tilt_speed_trajectory")
            errors = data["conf"]["tilt_error"]
            plot.data_wrt_episode_std_quantile(
                ax,
                data["result"]["tilt_error"] * DEG_TO_PX,
                ylim=ylim,
                xlabel="#Iteration (tilt)",
            )
            ax.axhline(1, color="k", linestyle="--", alpha=0.5)
            ax.axhline(-1, color="k", linestyle="--", alpha=0.5)

            ax = fig.add_subplot(143)
            data = self.data_by_name("vergence_trajectory")
            errors = data["conf"]["vergence_error"]
            plot.data_wrt_episode_std_quantile(
                ax,
                data["result"]["vergence_error"] * DEG_TO_PX,
                ylim=ylim,
                xlabel="#Iteration (vergence)",
            )
            ax.axhline(1, color="k", linestyle="--", alpha=0.5)
            ax.axhline(-1, color="k", linestyle="--", alpha=0.5)

            ax = fig.add_subplot(144)
            data = self.data_by_name("cyclo_trajectory")
            errors = data["conf"]["cyclo_pos"]
            plot.data_wrt_episode_std_quantile(
                ax,
                data["result"]["cyclo_pos"],
                ylim=[-6, 6],
                xlabel="#Iteration (cyclo)",
            )

    def plot_critic_error(self, path, save=True):
        if self.missing_data("pan_speed_trajectory", "tilt_speed_trajectory", "vergence_trajectory", "cyclo_trajectory"):
            return
        N = 10
        with plot.FigureManager(path + "/critic_error.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("pan_speed_trajectory", dim0="conf.pan_error", sort_order="conf.stimulus")
            plot.critic_error_wrt_episode(
                ax,
                data["result"]["pan_return_estimates"][:N],
                data["result"]["recerr_magno"][:N],
                xlabel="pan",
                ylabel="Reward"
            )

            ax = fig.add_subplot(142)
            data = self.data_by_name("tilt_speed_trajectory", dim0="conf.tilt_error", sort_order="conf.stimulus")
            plot.critic_error_wrt_episode(
                ax,
                data["result"]["tilt_return_estimates"][:N],
                data["result"]["recerr_magno"][:N],
                xlabel="tilt",
            )

            ax = fig.add_subplot(143)
            data = self.data_by_name("vergence_trajectory", dim0="conf.vergence_error", sort_order="conf.stimulus")
            plot.critic_error_wrt_episode(
                ax,
                data["result"]["vergence_return_estimates"][:N],
                data["result"]["recerr_pavro"][:N],
                xlabel="vergence",
            )

            ax = fig.add_subplot(144)
            data = self.data_by_name("cyclo_trajectory", dim0="conf.cyclo_pos", sort_order="conf.stimulus")
            plot.critic_error_wrt_episode(
                ax,
                data["result"]["cyclo_return_estimates"][:N],
                data["result"]["recerr_pavro"][:N],
                xlabel="cyclo",
            )

    def plot_reward_wrt_delta_error(self, path, save=True):
        if self.missing_data("pan_speed_trajectory", "tilt_speed_trajectory", "vergence_trajectory", "cyclo_trajectory"):
            return
        with plot.FigureManager(path + "/reward.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("pan_speed_trajectory")
            abs_error = np.abs(data["result"]["pan_error"])
            plot.scatter_plot(
                ax,
                abs_error[..., :-1] - abs_error[..., 1:],
                (data["result"]["recerr_magno"][..., :-1] - data["result"]["recerr_magno"][..., 1:]) * 600,
                xlabel="delta pan error",
                ylabel="Reward"
            )

            ax = fig.add_subplot(142)
            data = self.data_by_name("tilt_speed_trajectory")
            abs_error = np.abs(data["result"]["tilt_error"])
            plot.scatter_plot(
                ax,
                abs_error[..., :-1] - abs_error[..., 1:],
                (data["result"]["recerr_magno"][..., :-1] - data["result"]["recerr_magno"][..., 1:]) * 600,
                xlabel="tilt",
            )

            ax = fig.add_subplot(143)
            data = self.data_by_name("vergence_trajectory")
            abs_error = np.abs(data["result"]["vergence_error"])
            plot.scatter_plot(
                ax,
                abs_error[..., :-1] - abs_error[..., 1:],
                (data["result"]["recerr_pavro"][..., :-1] - data["result"]["recerr_pavro"][..., 1:]) * 600,
                xlabel="vergence",
            )

            ax = fig.add_subplot(144)
            data = self.data_by_name("cyclo_trajectory")
            abs_error = np.abs(data["result"]["cyclo_pos"])
            plot.scatter_plot(
                ax,
                abs_error[..., :-1] - abs_error[..., 1:],
                (data["result"]["recerr_pavro"][..., :-1] - data["result"]["recerr_pavro"][..., 1:]) * 600,
                xlabel="cyclo",
            )

    def plot_critic_wrt_reward(self, path, save=True):
        if self.missing_data("pan_speed_trajectory", "tilt_speed_trajectory", "vergence_trajectory", "cyclo_trajectory"):
            return
        with plot.FigureManager(path + "/critic_reward.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("pan_speed_trajectory")
            abs_error = np.abs(data["result"]["pan_error"])
            plot.scatter_plot(
                ax,
                (data["result"]["recerr_magno"][..., :-1] - data["result"]["recerr_magno"][..., 1:]) * 600,
                data["result"]["pan_return_estimates"][..., :-1],
                xlabel="pan reward",
                ylabel="Critic"
            )

            ax = fig.add_subplot(142)
            data = self.data_by_name("tilt_speed_trajectory")
            abs_error = np.abs(data["result"]["tilt_error"])
            plot.scatter_plot(
                ax,
                (data["result"]["recerr_magno"][..., :-1] - data["result"]["recerr_magno"][..., 1:]) * 600,
                data["result"]["tilt_return_estimates"][..., :-1],
                xlabel="tilt",
            )

            ax = fig.add_subplot(143)
            data = self.data_by_name("vergence_trajectory")
            abs_error = np.abs(data["result"]["vergence_error"])
            plot.scatter_plot(
                ax,
                (data["result"]["recerr_pavro"][..., :-1] - data["result"]["recerr_pavro"][..., 1:]) * 600,
                data["result"]["vergence_return_estimates"][..., :-1],
                xlabel="vergence",
            )

            ax = fig.add_subplot(144)
            data = self.data_by_name("cyclo_trajectory")
            abs_error = np.abs(data["result"]["cyclo_pos"])
            plot.scatter_plot(
                ax,
                (data["result"]["recerr_pavro"][..., :-1] - data["result"]["recerr_pavro"][..., 1:]) * 600,
                data["result"]["cyclo_return_estimates"][..., :-1],
                xlabel="cyclo",
            )

    def plot_critic_wrt_delta_error(self, path, save=True):
        if self.missing_data("pan_speed_trajectory", "tilt_speed_trajectory", "vergence_trajectory", "cyclo_trajectory"):
            return
        with plot.FigureManager(path + "/critic.png", save=save) as fig:
            ax = fig.add_subplot(141)
            data = self.data_by_name("pan_speed_trajectory")
            abs_error = np.abs(data["result"]["pan_error"])
            plot.scatter_plot(
                ax,
                data["result"]["pan_return_estimates"][..., :-1],
                abs_error[..., :-1] - abs_error[..., 1:],
                xlabel="delta pan error",
                ylabel="Predicted return"
            )

            ax = fig.add_subplot(142)
            data = self.data_by_name("tilt_speed_trajectory")
            abs_error = np.abs(data["result"]["tilt_error"])
            plot.scatter_plot(
                ax,
                data["result"]["tilt_return_estimates"][..., :-1],
                abs_error[..., :-1] - abs_error[..., 1:],
                xlabel="tilt",
            )

            ax = fig.add_subplot(143)
            data = self.data_by_name("vergence_trajectory")
            abs_error = np.abs(data["result"]["vergence_error"])
            plot.scatter_plot(
                ax,
                data["result"]["vergence_return_estimates"][..., :-1],
                abs_error[..., :-1] - abs_error[..., 1:],
                xlabel="vergence",
            )

            ax = fig.add_subplot(144)
            data = self.data_by_name("cyclo_trajectory")
            abs_error = np.abs(data["result"]["cyclo_pos"])
            plot.scatter_plot(
                ax,
                data["result"]["cyclo_return_estimates"][..., :-1],
                abs_error[..., :-1] - abs_error[..., 1:],
                xlabel="cyclo",
            )

    def plot(self, path, save=True):
        os.makedirs(path, exist_ok=True)
        self.plot_recerr_wrt_error(path, save=save)
        self.plot_predicted_recerr_wrt_error(path, save=save)
        self.plot_action_wrt_error(path, save=save)
        self.plot_action_wrt_error_individual(path, save=save)
        self.plot_abs_error_in_episode(path, save=save)
        self.plot_error_in_episode(path, save=save)
        self.plot_critic_error(path, save=save)
        self.plot_reward_wrt_delta_error(path, save=save)
        self.plot_critic_wrt_delta_error(path, save=save)
        self.plot_critic_wrt_reward(path, save=save)

    def get_performance(self, percentiles):
        tmp = {
            'vergence_trajectory': 'vergence_error',
            'cyclo_trajectory': 'cyclo_pos',
            'pan_speed_trajectory': 'pan_error',
            'tilt_speed_trajectory': 'tilt_error',
        }
        data = {
            value: np.abs(self.data_by_name(key)['result'][value][:, 9])
            for key, value in tmp.items()
        }
        return {
            key: tuple(
                np.percentile(value, q)
                for q in percentiles
            )
            for key, value in data.items()
        }


def test_case(stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error, n_iterations):
    return np.array((
        int(stimulus),
        float(object_distance),
        float(vergence_error),
        float(cyclo_pos),
        float(pan_error),
        float(tilt_error),
        int(n_iterations)), dtype=dttest_case)


def test_cases_between(stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error, n_iterations):
    total_length = np.product([
        len(x)
        for x in [stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error]
    ])
    ret = np.zeros(total_length, dtype=dttest_case)
    for i, case in enumerate(starmap(test_case, product(stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error, [n_iterations]))):
        ret[i] = case
    return ret


def test_data(stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error, n_iterations):
    test_case_arr = test_case(stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error, n_iterations)
    test_data_arr = np.zeros(1, dtype=dttest_data(n_iterations))
    test_data_arr["conf"] = test_case_arr
    return test_data_arr


def test_data_between(stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error, n_iterations):
    total_length = np.product([
        len(x)
        for x in [stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error]
    ])
    ret = np.zeros(total_length, dtype=dttest_data(n_iterations))
    for i, data in enumerate(starmap(test_data, product(stimulus, object_distance, vergence_error, cyclo_pos, pan_error, tilt_error, [n_iterations]))):
        ret[i] = data
    return ret


def generate_default_test_conf():
    errors = [90 / 320 * i for i in [-0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]]
    bound_in_px = 8
    cyclo_bound_in_deg = 5
    for distance in [2, 4, 6]:
        test_conf = TestDataContainer(
            stimulus=range(20),
            object_distance=[distance],
            pan_error=errors,
            tilt_error=errors,
            vergence_error=errors,
            cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
            n_iterations=20,
            name="default_at_{}m".format(distance)
        )
        test_conf.add_vergence_trajectory()
        test_conf.add_cyclo_trajectory()
        test_conf.add_speed_trajectory("tilt")
        test_conf.add_speed_trajectory("pan")
        test_conf.add_wrt_vergence_error(bound_in_px)
        test_conf.add_wrt_cyclo_pos(cyclo_bound_in_deg)
        test_conf.add_wrt_speed_error("tilt", bound_in_px)
        test_conf.add_wrt_speed_error("pan", bound_in_px)
        test_conf.dump("../config/test_conf/")


def generate_advanced_test_conf():
    errors = [90 / 320 * i for i in [-0.25, -0.5, -0.75, -1, -1.25, -1.5, -1.75, -2, -2.25, -2.5, -2.75, -3, -3.25, -3.5, -3.75, -4, -4.25, -4.5, -4.75, -5, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5]]
    bound_in_px = 8
    cyclo_bound_in_deg = 7.16
    object_distance = [1, 2, 3, 4, 5]
    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=object_distance,
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 20).astype(np.float32),
        n_iterations=10,
        name="advanced_testing",
    )
    test_conf.add_vergence_trajectory()
    test_conf.add_cyclo_trajectory()
    test_conf.add_speed_trajectory("tilt")
    test_conf.add_speed_trajectory("pan")
    test_conf.add_wrt_vergence_error(bound_in_px)
    test_conf.add_wrt_cyclo_pos(cyclo_bound_in_deg)
    test_conf.add_wrt_speed_error("tilt", bound_in_px)
    test_conf.add_wrt_speed_error("pan", bound_in_px)
    test_conf.dump("../config/test_conf/")


def generate_wide_error_range():
    errors = [90 / 320 * i for i in [-0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]]
    bound_in_px = 24
    cyclo_bound_in_deg = 15
    distance = 2

    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=[distance],
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
        n_iterations=20,
        name="wrt_each_joint_at_2m_wide_error_range"
    )
    test_conf.add_wrt_speed_error("pan", bound_in_px)
    test_conf.add_wrt_speed_error("tilt", bound_in_px)
    test_conf.add_wrt_vergence_error(bound_in_px)
    test_conf.add_wrt_cyclo_pos(cyclo_bound_in_deg)
    test_conf.dump("../config/test_conf/")

    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=[distance],
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
        n_iterations=20,
        name="wrt_pan_tilt_error_only_at_2m_wide_error_range"
    )
    test_conf.add_wrt_speed_error("pan", bound_in_px)
    test_conf.add_wrt_speed_error("tilt", bound_in_px)
    test_conf.dump("../config/test_conf/")

    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=[distance],
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
        n_iterations=20,
        name="wrt_vergence_error_only_at_2m_wide_error_range"
    )
    test_conf.add_wrt_vergence_error(bound_in_px)
    test_conf.dump("../config/test_conf/")

    test_conf = TestDataContainer(
        stimulus=range(20),
        object_distance=[distance],
        pan_error=errors,
        tilt_error=errors,
        vergence_error=errors,
        cyclo_pos=np.linspace(-cyclo_bound_in_deg, cyclo_bound_in_deg, 8).astype(np.float32),
        n_iterations=20,
        name="wrt_cyclo_error_only_at_2m_wide_error_range"
    )
    test_conf.add_wrt_cyclo_pos(cyclo_bound_in_deg)
    test_conf.dump("../config/test_conf/")

if __name__ == "__main__":
    generate_default_test_conf()
    generate_advanced_test_conf()
