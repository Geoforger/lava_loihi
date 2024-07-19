import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_pooling_dim(input_dim, kernel_dim, stride, order):
    """
    Function to calculate the size of a dimension after pooling

    Parameters
    ----------

    input_dim (int): Size of the dimension that is being pooled
    kernel_dim (int): Size of the pooling kernel along the same dimension
    stride (int): Stride of the pooling operation
    order (int): Number of pooling operations being performed
    """
    return int(((input_dim - kernel_dim) / stride) + (1 * order))


def heatmap_lava(data, orig_res, pooling_kernel, stride, order):
    """
    Function to plot lava events as a 2d heatmap of a given resolution

    Parameters
    ----------

    data (numpy array): Numpy array containing n timesteps of output data after pooling
    orig_res (tuple): Shape of the original data
    pooling_kernel (tuple): Shape of the pooling kernel used
    stride (int): Stride of the pooling operation
    order (int): Number of pooling operations performed
    """
    output_y = calculate_pooling_dim(orig_res[0], pooling_kernel[0], stride, order)
    output_x = calculate_pooling_dim(orig_res[1], pooling_kernel[1], stride, order)
    output_res = (output_y, output_x)

    neurons, num_ts = data.shape
    heat_map = np.zeros(output_res)

    # x = ts
    # y = neuron ID
    flat_data = np.zeros(neurons)

    for neuron in range(neurons):
        flat_data[neuron] = np.count_nonzero(data[neuron])

    heat_map = np.reshape(flat_data, output_res)
    ax = sns.heatmap(heat_map, linewidth=0.0)

    return ax


def raster_plot(data, title="Raster Plot",shaded_area=0, display=False, save=False, path=""):
    """
    Function to plot lava events as a raster plot

    Parameters
    ----------

    data (numpy array): Numpy array containing n timesteps of data
    """
    spikes_data = data.flatten()
    num_neurons = len(spikes_data)
    neuron_idx = 0
    final_spike = 0

    for spike_train in spikes_data:
        if spike_train != []:
            y = np.ones_like(spike_train) * neuron_idx
            plt.plot(spike_train, y, "k|", markersize=0.7)

            # Find final spike in the data for drawing the shaded region
            if max(spike_train) > final_spike:
                final_spike = max(spike_train)

        neuron_idx += 1

    fig = plt.figure(1, figsize=(5, 5))
    plt.ylabel("Neuron Idx")
    plt.xlabel("Time (ms)")
    plt.title(title)
    plt.setp(plt.gca().get_xticklabels(), visible=True)
    plt.tick_params(direction="in", which="minor", length=5, bottom=True, top=False)
    plt.tick_params(direction="in", which="major", length=8, bottom=True, top=False)
    plt.minorticks_on()
    plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    plt.yticks(np.arange(0, num_neurons + 500, 500))
    plt.xlim([-50, final_spike])
    plt.ylim([0, num_neurons + 500])

    if shaded_area != 0:
        # Place text in the non shaded area to show what is valid data
        valid_centre = (plt.xlim()[0] + shaded_area) / 2
        cropped_centre = (plt.xlim()[1] + shaded_area) / 2
        plt.text(valid_centre, num_neurons + 125, "Valid Data", color="black", ha="center")
        plt.text(cropped_centre, num_neurons + 125, "Cropped Data", color="black", ha="center")

        # Shade the area that is cropped
        plt.axvspan(shaded_area, final_spike, color="grey", alpha=0.3, label="Temporal Crop")

        # Add a vertical line at the given point (optional)
        plt.axvline(shaded_area, color="red", linestyle="--", label="Temporal Crop")

    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    if display:
        plt.show()

    plt.close()


def plot_raster_from_array(data, title="Raster Plot", shaded_area=0, display=False, save=False, path=""):
    """
    Plot a raster plot from a numpy array of 0s and 1s.

    Parameters:
    - data: 2D numpy array of shape (n_neurons, n_time_points)
    - title: Title of the plot
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    """
    n_neurons, n_time_points = data.shape

    plt.figure(figsize=(10, 6))

    for neuron in range(n_neurons):
        spike_times = np.where(data[neuron] == 1)[0]
        plt.vlines(spike_times, neuron + 0.5, neuron + 1.5)

    # plt.title(title)
    # plt.setp(plt.gca().get_xticklabels(), visible=True)
    # plt.tick_params(direction="in", which="minor", length=5, bottom=True, top=False)
    # plt.tick_params(direction="in", which="major", length=8, bottom=True, top=False)
    # plt.minorticks_on()
    # plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    # plt.xlabel("Timestep (ms)")
    # plt.ylabel("Neuron Idx")
    # plt.yticks(range(1, n_neurons + 1))
    # plt.ylim(0.5, n_neurons + 0.5)

    # if shaded_area != 0:
    #     # Place text in the non shaded area to show what is valid data
    #     valid_centre = (plt.xlim()[0] + shaded_area) / 2
    #     cropped_centre = (plt.xlim()[1] + shaded_area) / 2
    #     plt.text(valid_centre, n_neurons + 125, "Valid Data", color="black", ha="center")
    #     plt.text(cropped_centre, n_neurons + 125, "Cropped Data", color="black", ha="center")

    #     # Shade the area that is cropped
    #     plt.axvspan(shaded_area, n_time_points, color="grey", alpha=0.3, label="Temporal Crop")

    #     # Add a vertical line at the given point (optional)
    #     plt.axvline(shaded_area, color="red", linestyle="--", label="Temporal Crop")

    # if save:
    #     plt.savefig(path, dpi=300, bbox_inches="tight")
    # if display:
    #     plt.show()

    # plt.close()
