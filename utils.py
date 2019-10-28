import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def save_plot(x_values, y_values, title, save_path):
    fig, ax = plt.subplots()
    if len(x_values) > 0:
        ax.set(xlabel="Timesteps", ylabel="Score", title=title)
        ax.grid()

        poly = np.polyfit(x_values, y_values, 5)
        poly_y = np.poly1d(poly)(x_values)
        plt.plot(x_values, poly_y)

        fig.savefig(save_path)


def _find_available_dir_name(directory, subdir_name, n_call=1):
    # The folder aleady exists, we need to increment the
    # n_call until we find an available folder name
    new_subdir_name = f"{subdir_name}_{n_call}"
    path = os.path.join(directory, new_subdir_name)
    if os.path.exists(path):
        return _find_available_dir_name(directory, subdir_name, n_call=n_call + 1)
    else:
        return new_subdir_name


def check_subfolder_availability(directory, subdir_name):
    path = os.path.join(directory, subdir_name)
    if os.path.exists(path):
        return _find_available_dir_name(directory, subdir_name)
    else:
        return subdir_name
