import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sb
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import glob
from ast import literal_eval
import sys
sys.path.append("..")
from lava_loihi.utils.utils import nums_from_string

plt.rcParams.update(
    {
        "figure.figsize": (10, 10),  # (3.5,2.5),
        "font.size": 18,  # Set the font size
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        # "axes.labelsize": 8,  # TODO: Find appropriate values - Started @ 20, reduced by 4
        # "axes.titlesize": 8,
        # "legend.fontsize": 8,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "lines.linewidth": 4,
        "savefig.dpi": 300,
        "lines.markersize": 5,
        # "xtick.bottom": False,
        # "xtick.labelbottom": False,
        # "ytick.left": False,
        # "ytick.labelleft": False,
    }
)

def main():
    dataset_folder = (
        "/media/farscope2/T7 Shield/Neuromorphic Data/George/all_speeds_sorted/"
    )
    test_folder = "/home/farscope2/Documents/PhD/lava_loihi/data/speed_tests/speed_test_1721222363/"
    plot_folder = "/home/farscope2/Documents/PhD/lava_loihi/data/plots/"
    spikes_folder = f"{test_folder}/spikes/"

    # Import test meta params and confusion stats
    dataset_meta = pd.read_csv(f"{dataset_folder}/meta.csv", index_col=0).T
    test_stats = pd.read_csv(f"{test_folder}/output_labels.csv")

    # Create confusion matrix from data
    textures = dataset_meta["Textures"].iloc[0]
    textures = literal_eval(textures)
    real_labels = test_stats["Labels"].to_list()
    pred_labels = test_stats["Predictions"].to_list()
    speed_labels = test_stats["Speeds"].to_list()
    depth_labels = test_stats["Depths"].to_list()
    depths = [str(x) for x in np.unique(depth_labels)]

    cnf_matrix = confusion_matrix(real_labels, pred_labels, normalize="true")
    conf_mat_norm = np.around(
        cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis],
        decimals=2,
    )
    disp = ConfusionMatrixDisplay(conf_mat_norm)
    sb.heatmap(
        conf_mat_norm,
        annot=True,
        fmt="g",
        cmap="viridis",
        xticklabels=textures,
        yticklabels=textures,
        cbar=False
    )
    plt.title("Overall Testing Confusion Matrix")
    plt.xlabel("Textures")
    plt.ylabel("Textures")
    plt.xticks(
        np.arange(len(textures)),
        textures,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    plt.savefig(
        f"{plot_folder}/speed_test_confusion_matrix.png",
        dpi=600,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()

    # Analyse average accuracy at each speed for each texture
    testing_stats = pd.DataFrame({"idx": np.arange(len(real_labels)),
                                  "Label": real_labels,
                                  "Output Label": pred_labels,
                                  "Speed": speed_labels,
                                  "Depth": depth_labels
                                }, columns=["idx", "Label", "Output Label", "Speed", "Depth"])

    testing_stats["Label"] = testing_stats["Label"].apply(
        lambda x: textures[x]
    )
    testing_stats["Output Label"] = testing_stats["Output Label"].apply(
        lambda x: textures[x]
    )

    # Group by speed and texture
    label_speed_accuracy = (
        testing_stats.groupby(["Speed", "Label"])
        .apply(lambda x: (x["Label"] == x["Output Label"]).mean())
        .reset_index(name="Average Accuracy")
    )
    print(label_speed_accuracy)

    # Line plot for each unique label
    plt.figure(figsize=(12, 6))
    for label in textures:
        label_data = label_speed_accuracy[label_speed_accuracy["Label"] == label]
        plt.plot(
            label_data["Speed"], label_data["Average Accuracy"], label=f"{label}"
        )

    plt.xlabel("Speed (mm/s)")
    plt.ylabel("Mean Accuracy")
    plt.title("Mean Accuracy by Speed for Each Texture")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_folder}/speed_test_accuracy_speed.png", dpi=600)#, bbox_inches="tight")
    plt.show()
    plt.close()

    # Calculate the accuracy for each speed (averaged across all textures)
    speed_accuracy = testing_stats.groupby("Speed").apply(lambda x: (x["Label"] == x["Output Label"]).mean()).reset_index(name="Average Accuracy")

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(speed_accuracy["Speed"], speed_accuracy["Average Accuracy"], marker='o')

    plt.xlabel("Speed (mm/s)")
    plt.ylabel("Mean Accuracy")
    plt.title("Mean Accuracy by Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_folder}/speed_test_accuracy_overall_speed.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    # Group by depth and texture
    label_depth_accuracy = (
        testing_stats.groupby(["Depth", "Label"])
        .apply(lambda x: (x["Label"] == x["Output Label"]).mean())
        .reset_index(name="Average Accuracy")
    )
    label_depth_accuracy["Depth"] = label_depth_accuracy["Depth"].astype(float)
    print(label_depth_accuracy)

    # Line plot for each unique label
    plt.figure(figsize=(12, 6))
    for label in textures:
        label_data = label_depth_accuracy[label_depth_accuracy["Label"] == label]
        plt.plot(label_data["Depth"], label_data["Average Accuracy"], label=f"{label}")

    plt.xlabel("Depth (mm)")
    # plt.xticks(np.arange(len(depths)), depths)
    plt.ylabel("Mean Accuracy")
    plt.title("Mean Accuracy by Depth for Each Texture")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"{plot_folder}/speed_test_accuracy_depth.png", dpi=600, bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # Calculate the accuracy for each speed (averaged across all textures)
    depth_accuracy = (
        testing_stats.groupby("Depth")
        .apply(lambda x: (x["Label"] == x["Output Label"]).mean())
        .reset_index(name="Average Accuracy")
    )

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(depth_accuracy["Depth"], depth_accuracy["Average Accuracy"], marker="o")

    plt.xlabel("Depth (mm)")
    plt.ylabel("Mean Accuracy")
    plt.title("Mean Accuracy by Depth")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"{plot_folder}/speed_test_accuracy_overall_depth.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


# Analyse confidence over time
# TODO:
# - Over all tests @ each timestep:
#   - Find average max_spike
#   - Find average total_spikes
#   - Find average confidence
#   - Find average accuracy
# files = glob.glob(f"{spikes_folder}/*.npy")
# num_files = len(files)
# sample_length = dataset_meta["cuttoff"].iloc[1]
# depth = 1.5

# max_spikes = np.zeros((num_files, sample_length), dtype=np.int8)
# total_spikes = np.zeros_like(max_spikes)
# confidence = np.zeros((num_files, sample_length), dtype=np.float32)
# output_label = np.zeros((num_files, sample_length), dtype=np.int8)

# for idx, file in enumerate(files):
#     n = nums_from_string(file)
#     speed = n[-1]
#     label = n[-2]

#     sample = np.load(file)

#     # Cumulative spike count, max/total spikes and confidence per ts
#     sample_count = np.cumsum(sample, axis=1)
#     sample_max = np.max(sample_count, axis=0)
#     sample_total = np.sum(sample_count, axis=0)
#     sample_confidence = sample_max / sample_total

#     # Classification at each step

#     #

#     # Input data for analysis
#     data = {
#         "Sample": idx,
#         "Depth (mm)": depth,
#         "Speed (mm/s)": speed,
#         "Label": label,
#         "Classification":
#     }


if __name__ == "__main__":
    main()
