from test_data import TestDataContainer, DEG_TO_PX
import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile


def plot(ax, data, x_name, y_name, x_scale=DEG_TO_PX, label='std={}px'):
    color_A = np.array((150, 246, 255)) / 255
    color_B = np.array((  5,   0, 107)) / 255
    colors = [x * color_B + (1 - x) * color_A for x in np.linspace(0, 1, 6)]
    for color, (error, d) in zip(colors, data.items()):
        ax.plot(
            d["result"][x_name][0] * x_scale,
            np.mean(d["result"][y_name], axis=0),
            label=label.format(error=error),
            color=color,
        )


def plot_individual(ax, data, x_name, y_name, error, x_scale=DEG_TO_PX):
    # color_A = np.array((150, 246, 255)) / 255
    # color_B = np.array((  5,   0, 107)) / 255
    # colors = [x * color_B + (1 - x) * color_A for x in np.linspace(0, 1, 6)]
    for x, y in zip(data[error]["result"][x_name], data[error]["result"][y_name]):
        ax.plot(x * x_scale, y, color='b', alpha=0.7, linewidth=0.5)
    ax.plot(
        data[error]["result"][x_name][0] * x_scale,
        np.mean(data[error]["result"][y_name], axis=0),
        label='mean',
        color='r',
        linewidth=1.5,
    )


def plot_sum(ax, fine_data, coarse_data, x_name, y_name, x_scale=DEG_TO_PX, label='std={}px'):
    color_A = np.array((150, 246, 255)) / 255
    color_B = np.array((  5,   0, 107)) / 255
    colors = [x * color_B + (1 - x) * color_A for x in np.linspace(0, 1, 6)]
    for color, (error, fine_d) in zip(colors, fine_data.items()):
        coarse_d = coarse_data[error]
        ax.plot(
            fine_d["result"][x_name][0] * x_scale,
            np.mean((fine_d["result"][y_name] + coarse_d["result"][y_name]), axis=0) / 2,
            label=label.format(error=error),
            color=color,
        )


def plot_v_shapes(fig, vergence_paths, pan_tilt_paths, cyclo_paths, suptitle, ylabel):
    pan_tilt_containers = {error: TestDataContainer.load(path) for error, path in pan_tilt_paths.items() if isfile(path)}
    pan_data = {error: container.data_by_name("wrt_pan_error", dim0="conf.stimulus", sort_order="conf.pan_error") for error, container in pan_tilt_containers.items()}
    tilt_data = {error: container.data_by_name("wrt_tilt_error", dim0="conf.stimulus", sort_order="conf.tilt_error") for error, container in pan_tilt_containers.items()}
    vergence_containers = {error: TestDataContainer.load(path) for error, path in vergence_paths.items() if isfile(path)}
    vergence_data = {error: container.data_by_name("wrt_vergence_error", dim0="conf.stimulus", sort_order="conf.vergence_error") for error, container in vergence_containers.items()}
    cyclo_containers = {error: TestDataContainer.load(path) for error, path in cyclo_paths.items() if isfile(path)}
    cyclo_data = {error: container.data_by_name("wrt_cyclo_pos", dim0="conf.stimulus", sort_order="conf.cyclo_pos") for error, container in cyclo_containers.items()}

    ax = fig.add_subplot(141)
    plot(ax, pan_data, "pan_error", "recerr_magno", label=r'$\sigma={error}\mathrm{{px.it}}^{{-1}}$')
    ax.set_ylabel(ylabel)
    ax.set_xlabel("pan error $e_p$ ($\mathrm{{px.it}}^{{-1}}$)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(142)
    plot(ax, tilt_data, "tilt_error", "recerr_magno", label=r'$\sigma={error}\mathrm{{px.it}}^{{-1}}$')
    ax.set_xlabel("tilt error $e_t$ ($\mathrm{{px.it}}^{{-1}}$)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(143)
    plot(ax, vergence_data, "vergence_error", "recerr_pavro", label=r'$\sigma={error}$px')
    ax.set_xlabel("vergence error $e_v$ (px)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(144)
    plot(ax, cyclo_data, "cyclo_pos", "recerr_pavro", x_scale=1, label=r'$\sigma={error}$deg')
    ax.set_xlabel("cyclovergence error $e_c$ (deg)")
    ax.legend(prop={'size': 5})
    fig.tight_layout()
    # fig.suptitle(suptitle)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_v_shapes_individual(fig, vergence_paths, pan_tilt_paths, cyclo_paths, suptitle, ylabel, error):
    pan_tilt_containers = {error: TestDataContainer.load(path) for error, path in pan_tilt_paths.items() if isfile(path)}
    pan_data = {error: container.data_by_name("wrt_pan_error", dim0="conf.stimulus", sort_order="conf.pan_error") for error, container in pan_tilt_containers.items()}
    tilt_data = {error: container.data_by_name("wrt_tilt_error", dim0="conf.stimulus", sort_order="conf.tilt_error") for error, container in pan_tilt_containers.items()}
    vergence_containers = {error: TestDataContainer.load(path) for error, path in vergence_paths.items() if isfile(path)}
    vergence_data = {error: container.data_by_name("wrt_vergence_error", dim0="conf.stimulus", sort_order="conf.vergence_error") for error, container in vergence_containers.items()}
    cyclo_containers = {error: TestDataContainer.load(path) for error, path in cyclo_paths.items() if isfile(path)}
    cyclo_data = {error: container.data_by_name("wrt_cyclo_pos", dim0="conf.stimulus", sort_order="conf.cyclo_pos") for error, container in cyclo_containers.items()}

    ax = fig.add_subplot(141)
    plot_individual(ax, pan_data, "pan_error", "recerr_magno", error)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("pan error $e_p$ ($\mathrm{{px.it}}^{{-1}}$)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(142)
    plot_individual(ax, tilt_data, "tilt_error", "recerr_magno", error)
    ax.set_xlabel("tilt error $e_t$ ($\mathrm{{px.it}}^{{-1}}$)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(143)
    plot_individual(ax, vergence_data, "vergence_error", "recerr_pavro", error)
    ax.set_xlabel("vergence error $e_v$ (px)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(144)
    plot_individual(ax, cyclo_data, "cyclo_pos", "recerr_pavro", error, x_scale=1)
    ax.set_xlabel("cyclovergence error $e_c$ (deg)")
    ax.legend(prop={'size': 5})
    fig.tight_layout()
    # fig.suptitle(suptitle)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_v_shapes_sum(fig,
        fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths,
        coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths,
        suptitle, ylabel):
    fine_pan_tilt_containers = {error: TestDataContainer.load(path) for error, path in fine_pan_tilt_paths.items() if isfile(path)}
    fine_pan_data = {error: container.data_by_name("wrt_pan_error", dim0="conf.stimulus", sort_order="conf.pan_error") for error, container in fine_pan_tilt_containers.items()}
    fine_tilt_data = {error: container.data_by_name("wrt_tilt_error", dim0="conf.stimulus", sort_order="conf.tilt_error") for error, container in fine_pan_tilt_containers.items()}
    fine_vergence_containers = {error: TestDataContainer.load(path) for error, path in fine_vergence_paths.items() if isfile(path)}
    fine_vergence_data = {error: container.data_by_name("wrt_vergence_error", dim0="conf.stimulus", sort_order="conf.vergence_error") for error, container in fine_vergence_containers.items()}
    fine_cyclo_containers = {error: TestDataContainer.load(path) for error, path in fine_cyclo_paths.items() if isfile(path)}
    fine_cyclo_data = {error: container.data_by_name("wrt_cyclo_pos", dim0="conf.stimulus", sort_order="conf.cyclo_pos") for error, container in fine_cyclo_containers.items()}
    coarse_pan_tilt_containers = {error: TestDataContainer.load(path) for error, path in coarse_pan_tilt_paths.items() if isfile(path)}
    coarse_pan_data = {error: container.data_by_name("wrt_pan_error", dim0="conf.stimulus", sort_order="conf.pan_error") for error, container in coarse_pan_tilt_containers.items()}
    coarse_tilt_data = {error: container.data_by_name("wrt_tilt_error", dim0="conf.stimulus", sort_order="conf.tilt_error") for error, container in coarse_pan_tilt_containers.items()}
    coarse_vergence_containers = {error: TestDataContainer.load(path) for error, path in coarse_vergence_paths.items() if isfile(path)}
    coarse_vergence_data = {error: container.data_by_name("wrt_vergence_error", dim0="conf.stimulus", sort_order="conf.vergence_error") for error, container in coarse_vergence_containers.items()}
    coarse_cyclo_containers = {error: TestDataContainer.load(path) for error, path in coarse_cyclo_paths.items() if isfile(path)}
    coarse_cyclo_data = {error: container.data_by_name("wrt_cyclo_pos", dim0="conf.stimulus", sort_order="conf.cyclo_pos") for error, container in coarse_cyclo_containers.items()}

    ax = fig.add_subplot(141)
    plot_sum(ax, fine_pan_data, coarse_pan_data, "pan_error", "recerr_magno", label=r'$\sigma={error}\mathrm{{px.it}}^{{-1}}$')
    ax.set_ylabel(ylabel)
    ax.set_xlabel("pan error $e_p$ ($\mathrm{{px.it}}^{{-1}}$)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(142)
    plot_sum(ax, fine_tilt_data, coarse_tilt_data, "tilt_error", "recerr_magno", label=r'$\sigma={error}\mathrm{{px.it}}^{{-1}}$')
    ax.set_xlabel("tilt error $e_t$ ($\mathrm{{px.it}}^{{-1}}$)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(143)
    plot_sum(ax, fine_vergence_data, coarse_vergence_data, "vergence_error", "recerr_pavro", label=r'$\sigma={error}$px')
    ax.set_xlabel("vergence error $e_v$ (px)")
    ax.legend(prop={'size': 5})
    ax = fig.add_subplot(144)
    plot_sum(ax, fine_cyclo_data, coarse_cyclo_data, "cyclo_pos", "recerr_pavro", x_scale=1, label=r'$\sigma={error}$deg')
    ax.set_xlabel("cyclovergence error $e_c$ (deg)")
    ax.legend(prop={'size': 5})
    fig.tight_layout()
    # fig.suptitle(suptitle)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])


if __name__ == '__main__':
    save = True

    vergence_paths = {
        0: "../experiments/2021-10-02/14-04-42_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-02/14-06-53_train_from_10000samples_vergence_1_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_1_cyclo_0/tests/speed_0_vergence_1_cyclo_0.pkl",
        2: "../experiments/2021-10-02/14-07-14_train_from_10000samples_vergence_2_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_2_cyclo_0/tests/speed_0_vergence_2_cyclo_0.pkl",
        4: "../experiments/2021-10-02/14-07-36_train_from_10000samples_vergence_4_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_4_cyclo_0/tests/speed_0_vergence_4_cyclo_0.pkl",
        8: "../experiments/2021-10-03/17-29-09_train_from_10000samples_vergence_8_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_8_cyclo_0/tests/speed_0_vergence_8_cyclo_0.pkl",
        16:"../experiments/2021-10-02/14-08-20_train_from_10000samples_vergence_16_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_16_cyclo_0/tests/speed_0_vergence_16_cyclo_0.pkl",
    }
    pan_tilt_paths = {
        0: "../experiments/2021-10-02/14-04-42_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-02/14-08-41_train_from_10000samples_vergence_0_speed_1_cyclo_0.dat_critic.False__name.speed_1_vergence_0_cyclo_0/tests/speed_1_vergence_0_cyclo_0.pkl",
        2: "../experiments/2021-10-02/14-09-03_train_from_10000samples_vergence_0_speed_2_cyclo_0.dat_critic.False__name.speed_2_vergence_0_cyclo_0/tests/speed_2_vergence_0_cyclo_0.pkl",
        4: "../experiments/2021-10-03/12-26-32_train_from_10000samples_vergence_0_speed_4_cyclo_0.dat_critic.False__name.speed_4_vergence_0_cyclo_0/tests/speed_4_vergence_0_cyclo_0.pkl",
        8: "../experiments/2021-10-02/14-09-47_train_from_10000samples_vergence_0_speed_8_cyclo_0.dat_critic.False__name.speed_8_vergence_0_cyclo_0/tests/speed_8_vergence_0_cyclo_0.pkl",
        16:"../experiments/2021-10-03/12-21-37_train_from_10000samples_vergence_0_speed_16_cyclo_0.dat_critic.False__name.speed_16_vergence_0_cyclo_0/tests/speed_16_vergence_0_cyclo_0.pkl",
    }
    cyclo_paths = {
        0: "../experiments/2021-10-02/14-04-42_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-02/14-05-04_train_from_10000samples_vergence_0_speed_0_cyclo_1.dat_critic.False__name.speed_0_vergence_0_cyclo_1/tests/speed_0_vergence_0_cyclo_1.pkl",
        2: "../experiments/2021-10-02/14-05-26_train_from_10000samples_vergence_0_speed_0_cyclo_2.dat_critic.False__name.speed_0_vergence_0_cyclo_2/tests/speed_0_vergence_0_cyclo_2.pkl",
        4: "../experiments/2021-10-02/14-05-47_train_from_10000samples_vergence_0_speed_0_cyclo_4.dat_critic.False__name.speed_0_vergence_0_cyclo_4/tests/speed_0_vergence_0_cyclo_4.pkl",
        8: "../experiments/2021-10-03/17-28-47_train_from_10000samples_vergence_0_speed_0_cyclo_8.dat_critic.False__name.speed_0_vergence_0_cyclo_8/tests/speed_0_vergence_0_cyclo_8.pkl",
        16:"../experiments/2021-10-02/14-06-31_train_from_10000samples_vergence_0_speed_0_cyclo_16.dat_critic.False__name.speed_0_vergence_0_cyclo_16/tests/speed_0_vergence_0_cyclo_16.pkl",
    }

    fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    plot_v_shapes(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$")
    if save:
        fig.savefig("../tmp/vshape.pdf")
    else:
        plt.show()
    plt.close(fig)

    vergence_paths = {
        0: "../experiments/2021-10-03/17-34-01_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-03/17-48-26_train_from_10000samples_vergence_1_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_1_cyclo_0/tests/speed_0_vergence_1_cyclo_0.pkl",
        2: "../experiments/2021-10-03/17-48-47_train_from_10000samples_vergence_2_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_2_cyclo_0/tests/speed_0_vergence_2_cyclo_0.pkl",
        4: "../experiments/2021-10-03/17-49-09_train_from_10000samples_vergence_4_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_4_cyclo_0/tests/speed_0_vergence_4_cyclo_0.pkl",
        8: "../experiments/2021-10-03/17-49-30_train_from_10000samples_vergence_8_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_8_cyclo_0/tests/speed_0_vergence_8_cyclo_0.pkl",
        16:"../experiments/2021-10-03/17-49-52_train_from_10000samples_vergence_16_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_16_cyclo_0/tests/speed_0_vergence_16_cyclo_0.pkl",
    }
    pan_tilt_paths = {
        0: "../experiments/2021-10-03/17-34-01_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-03/17-50-14_train_from_10000samples_vergence_0_speed_1_cyclo_0.dat_critic.False__name.speed_1_vergence_0_cyclo_0/tests/speed_1_vergence_0_cyclo_0.pkl",
        2: "../experiments/2021-10-03/17-50-36_train_from_10000samples_vergence_0_speed_2_cyclo_0.dat_critic.False__name.speed_2_vergence_0_cyclo_0/tests/speed_2_vergence_0_cyclo_0.pkl",
        4: "../experiments/2021-10-03/17-50-58_train_from_10000samples_vergence_0_speed_4_cyclo_0.dat_critic.False__name.speed_4_vergence_0_cyclo_0/tests/speed_4_vergence_0_cyclo_0.pkl",
        8: "../experiments/2021-10-03/17-51-20_train_from_10000samples_vergence_0_speed_8_cyclo_0.dat_critic.False__name.speed_8_vergence_0_cyclo_0/tests/speed_8_vergence_0_cyclo_0.pkl",
        16:"../experiments/2021-10-03/17-51-42_train_from_10000samples_vergence_0_speed_16_cyclo_0.dat_critic.False__name.speed_16_vergence_0_cyclo_0/tests/speed_16_vergence_0_cyclo_0.pkl",
    }
    cyclo_paths = {
        0: "../experiments/2021-10-03/17-34-01_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-03/17-46-38_train_from_10000samples_vergence_0_speed_0_cyclo_1.dat_critic.False__name.speed_0_vergence_0_cyclo_1/tests/speed_0_vergence_0_cyclo_1.pkl",
        2: "../experiments/2021-10-03/17-46-59_train_from_10000samples_vergence_0_speed_0_cyclo_2.dat_critic.False__name.speed_0_vergence_0_cyclo_2/tests/speed_0_vergence_0_cyclo_2.pkl",
        4: "../experiments/2021-10-03/17-47-21_train_from_10000samples_vergence_0_speed_0_cyclo_4.dat_critic.False__name.speed_0_vergence_0_cyclo_4/tests/speed_0_vergence_0_cyclo_4.pkl",
        8: "../experiments/2021-10-03/17-47-43_train_from_10000samples_vergence_0_speed_0_cyclo_8.dat_critic.False__name.speed_0_vergence_0_cyclo_8/tests/speed_0_vergence_0_cyclo_8.pkl",
        16:"../experiments/2021-10-03/17-48-04_train_from_10000samples_vergence_0_speed_0_cyclo_16.dat_critic.False__name.speed_0_vergence_0_cyclo_16/tests/speed_0_vergence_0_cyclo_16.pkl",
    }

    fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    plot_v_shapes(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$")
    if save:
        fig.savefig("../tmp/vshape_fine.pdf")
    else:
        plt.show()
    plt.close(fig)

    vergence_paths = {
        0: "../experiments/2021-10-03/17-52-03_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-03/17-54-13_train_from_10000samples_vergence_1_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_1_cyclo_0/tests/speed_0_vergence_1_cyclo_0.pkl",
        2: "../experiments/2021-10-03/17-54-34_train_from_10000samples_vergence_2_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_2_cyclo_0/tests/speed_0_vergence_2_cyclo_0.pkl",
        4: "../experiments/2021-10-03/17-54-56_train_from_10000samples_vergence_4_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_4_cyclo_0/tests/speed_0_vergence_4_cyclo_0.pkl",
        8: "../experiments/2021-10-03/17-55-18_train_from_10000samples_vergence_8_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_8_cyclo_0/tests/speed_0_vergence_8_cyclo_0.pkl",
        16:"../experiments/2021-10-03/17-55-39_train_from_10000samples_vergence_16_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_16_cyclo_0/tests/speed_0_vergence_16_cyclo_0.pkl",
    }
    pan_tilt_paths = {
        0: "../experiments/2021-10-03/17-52-03_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-03/17-56-01_train_from_10000samples_vergence_0_speed_1_cyclo_0.dat_critic.False__name.speed_1_vergence_0_cyclo_0/tests/speed_1_vergence_0_cyclo_0.pkl",
        2: "../experiments/2021-10-03/17-56-23_train_from_10000samples_vergence_0_speed_2_cyclo_0.dat_critic.False__name.speed_2_vergence_0_cyclo_0/tests/speed_2_vergence_0_cyclo_0.pkl",
        4: "../experiments/2021-10-03/17-56-44_train_from_10000samples_vergence_0_speed_4_cyclo_0.dat_critic.False__name.speed_4_vergence_0_cyclo_0/tests/speed_4_vergence_0_cyclo_0.pkl",
        8: "../experiments/2021-10-03/17-57-06_train_from_10000samples_vergence_0_speed_8_cyclo_0.dat_critic.False__name.speed_8_vergence_0_cyclo_0/tests/speed_8_vergence_0_cyclo_0.pkl",
        16:"../experiments/2021-10-03/17-57-28_train_from_10000samples_vergence_0_speed_16_cyclo_0.dat_critic.False__name.speed_16_vergence_0_cyclo_0/tests/speed_16_vergence_0_cyclo_0.pkl",
    }
    cyclo_paths = {
        0: "../experiments/2021-10-03/17-52-03_train_from_10000samples_vergence_0_speed_0_cyclo_0.dat_critic.False__name.speed_0_vergence_0_cyclo_0/tests/speed_0_vergence_0_cyclo_0.pkl",
        1: "../experiments/2021-10-03/17-52-25_train_from_10000samples_vergence_0_speed_0_cyclo_1.dat_critic.False__name.speed_0_vergence_0_cyclo_1/tests/speed_0_vergence_0_cyclo_1.pkl",
        2: "../experiments/2021-10-03/17-52-46_train_from_10000samples_vergence_0_speed_0_cyclo_2.dat_critic.False__name.speed_0_vergence_0_cyclo_2/tests/speed_0_vergence_0_cyclo_2.pkl",
        4: "../experiments/2021-10-03/17-53-08_train_from_10000samples_vergence_0_speed_0_cyclo_4.dat_critic.False__name.speed_0_vergence_0_cyclo_4/tests/speed_0_vergence_0_cyclo_4.pkl",
        8: "../experiments/2021-10-03/17-53-29_train_from_10000samples_vergence_0_speed_0_cyclo_8.dat_critic.False__name.speed_0_vergence_0_cyclo_8/tests/speed_0_vergence_0_cyclo_8.pkl",
        16:"../experiments/2021-10-03/17-53-51_train_from_10000samples_vergence_0_speed_0_cyclo_16.dat_critic.False__name.speed_0_vergence_0_cyclo_16/tests/speed_0_vergence_0_cyclo_16.pkl",
    }

    fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    plot_v_shapes(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$")
    if save:
        fig.savefig("../tmp/vshape_coarse.pdf")
    else:
        plt.show()
    plt.close(fig)

    # vergence_paths = {
    #     0: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$")
    # if save:
    #     fig.savefig("../tmp/vshape_1.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_1_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fine_vergence_paths = {
    #     0: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # fine_pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # fine_cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, "fine scale only", r"reconstruction error $l_\mathrm{{fine}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_fine_1.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, "fine scale only", r"reconstruction error $l_\mathrm{{fine}}$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_fine_1_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # coarse_vergence_paths = {
    #     0: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # coarse_pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # coarse_cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_coarse_1.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_coarse_1_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_sum(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_sum_1.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    #
    #
    # vergence_paths = {
    #     0: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$")
    # if save:
    #     fig.savefig("../tmp/vshape_2.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_2_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fine_vergence_paths = {
    #     0: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # fine_pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # fine_cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, "fine scale only", r"reconstruction error $l_\mathrm{{fine}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_fine_2.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, "fine scale only", r"reconstruction error $l_\mathrm{{fine}}$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_fine_2_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # coarse_vergence_paths = {
    #     0: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # coarse_pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # coarse_cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_coarse_2.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_coarse_2_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_sum(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_sum_2.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    #
    # ############################################################################
    # ##  WIDE RANGE  ############################################################
    # ############################################################################
    #
    #
    # vergence_paths = {
    #     0: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-41-09_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$")
    # if save:
    #     fig.savefig("../tmp/vshape_wide_range_1.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_wide_range_1_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fine_vergence_paths = {
    #     0: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # fine_pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # fine_cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-41-31_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, "fine scale only", r"reconstruction error $l_\mathrm{{fine}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_fine_wide_range_1.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, "fine scale only", r"reconstruction error $l_\mathrm{{fine}}$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_fine_wide_range_1_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # coarse_vergence_paths = {
    #     0: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # coarse_pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # coarse_cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-41-52_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_coarse_wide_range_1.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_coarse_wide_range_1_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_sum(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_sum_wide_range_1.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    #
    #
    # vergence_paths = {
    #     0: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-42-14_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$")
    # if save:
    #     fig.savefig("../tmp/vshape_wide_range_2.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, vergence_paths, pan_tilt_paths, cyclo_paths, "fine scale and coarse scale", r"reconstruction error $l$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_wide_range_2_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fine_vergence_paths = {
    #     0: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # fine_pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # fine_cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-42-36_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, "fine scale only", r"reconstruction error $l_\mathrm{{fine}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_fine_wide_range_2.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, "fine scale only", r"reconstruction error $l_\mathrm{{fine}}$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_fine_wide_range_2_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # coarse_vergence_paths = {
    #     0: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_1_speed_0_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_2_speed_0_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_4_speed_0_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_8_speed_0_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_16_speed_0_cyclo_0.pkl",
    # }
    # coarse_pan_tilt_paths = {
    #     0: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_1_cyclo_0.pkl",
    #     2: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_2_cyclo_0.pkl",
    #     4: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_4_cyclo_0.pkl",
    #     8: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_8_cyclo_0.pkl",
    #     16:"../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_16_cyclo_0.pkl",
    # }
    # coarse_cyclo_paths = {
    #     0: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_0.pkl",
    #     1: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_1.pkl",
    #     2: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_2.pkl",
    #     4: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_4.pkl",
    #     8: "../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_8.pkl",
    #     16:"../experiments/2021-09-23/20-42-57_train_from_None_critic.False__n_training_steps.50000/tests/tests/vergence_0_speed_0_cyclo_16.pkl",
    # }
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes(fig, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_coarse_wide_range_2.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_individual(fig, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$", error=4)
    # if save:
    #     fig.savefig("../tmp/vshape_coarse_wide_range_2_individual.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
    #
    # fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    # plot_v_shapes_sum(fig, fine_vergence_paths, fine_pan_tilt_paths, fine_cyclo_paths, coarse_vergence_paths, coarse_pan_tilt_paths, coarse_cyclo_paths, "coarse scale only", r"reconstruction error $l_\mathrm{{coarse}}$")
    # if save:
    #     fig.savefig("../tmp/vshape_sum_wide_range_2.pdf")
    # else:
    #     plt.show()
    # plt.close(fig)
