import numpy as np


def plot_performance(ax, indices, fill_data, median, percentiles, data_scale=320 / 90, show_y1=True):
    n_regions = len(fill_data)
    n_colors = len(fill_data) // 2
    color_A = np.array((150, 246, 255)) / 255
    color_B = np.array((  5,   0, 107)) / 255
    # color_A = np.array([1, 1, 1])
    # color_B = np.array([0.3, 0.3, 0.3])
    for i, (ymini, ymaxi) in enumerate(fill_data.values()):
        color_interpolation = np.abs(i - (n_regions - 1) / 2) * 2 / (n_regions - 1)
        color = color_A * color_interpolation + color_B * (1 - color_interpolation)
        if i >= n_colors:
            label = None
        elif i == n_colors - 1:
            label = f'[{percentiles[i]},{percentiles[n_regions - i]}]'
        else:
            label = f'[{percentiles[i]},{percentiles[i+1]}],[{percentiles[n_regions - i - 1]},{percentiles[n_regions - i]}]'
        ax.fill_between(
            indices,
            np.array(ymini) * data_scale,
            np.array(ymaxi) * data_scale,
            color=color,
            label=label,
        )
    ax.plot(indices, np.array(median) * data_scale, color='r', label='50')
    if show_y1:
        ax.axhline(1, color='k', linestyle='--')
    ax.set_ylim([-0.2, 10])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from test_data import TestDataContainer
    import sys
    import re
    import os

    path = sys.argv[1]
    plot_path = path + '/../plots/'
    test_files_to_index = {
        file: int(match.group(1))
        for file in os.listdir(path)
        if (match := re.match(r'.*_([0-9]+)\.pkl', file))
    }
    test_files = sorted(test_files_to_index, key=test_files_to_index.get)
    indices = sorted(test_files_to_index.values())
    test_data = [TestDataContainer.load(path + '/' + f) for f in test_files]
    # percentiles = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    percentiles = np.arange(5, 100, 5)
    results = [td.get_performance(percentiles) for td in test_data]
    m = np.argmax(percentiles == 50)
    median = {k: [x[k][m] for x in results] for k in results[0]}
    fill_data = {}
    for key in results[0]:
        fill_data[key] = {}
        for i, (p0, p1) in enumerate(zip(percentiles, percentiles[1:])):
            fill_data[key][(p0, p1)] = (
                [r[key][i + 0] for r in results],
                [r[key][i + 1] for r in results],
            )

    fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    ax = fig.add_subplot(141)
    plot_performance(ax, indices, fill_data['pan_error'], median['pan_error'], percentiles)
    ax.set_ylabel(
"""
absolute pan error $|e_p|$ ($\mathrm{{px.it}}^{{-1}}$)
absolute tilt error $|e_t|$ ($\mathrm{{px.it}}^{{-1}}$)
absolute vergence error $|e_v|$ (px)
absolute cyclovergence error $|e_c|$ (deg)
""")
    ax.set_xticks([0, 100000])
    ax.set_xticklabels(['0', '1e6'])
    ax.set_xlabel("#episode\n(pan)")
    ax = fig.add_subplot(142)
    plot_performance(ax, indices, fill_data['tilt_error'], median['tilt_error'], percentiles)
    ax.set_xlabel("#episode\n(tilt)")
    ax.set_xticks([0, 100000])
    ax.set_xticklabels(['0', '1e6'])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(143)
    plot_performance(ax, indices, fill_data['vergence_error'], median['vergence_error'], percentiles)
    ax.set_xlabel("#episode\n(vergence)")
    ax.set_xticks([0, 100000])
    ax.set_xticklabels(['0', '1e6'])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(144)
    plot_performance(ax, indices, fill_data['cyclo_pos'], median['cyclo_pos'], percentiles, data_scale=1, show_y1=False)
    ax.set_xlabel("#episode\n(cyclovergence)")
    ax.set_xticks([0, 100000])
    ax.set_xticklabels(['0', '1e6'])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.legend(prop={'size': 7})
    fig.tight_layout()
    # plt.show()
    fig.savefig(plot_path + "performances.pdf")
