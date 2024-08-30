import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    input_path = ""
    output_path = "../data/plots/"

    materials = [
        "Aluminium",
        "Acrylic",
        "MDF",
        "Plywood",
        "Fake Leather",
        "Denim",
        "Silicon",
        "Foam",
        "Satin",
        "Velvet",
        "Fake Fur",
        "Wool",
    ]

    results_frame = pd.read_csv(f"{input_path}/output_labels.csv")

    real_labels = results_frame["Labels"].to_list()
    pred_labels = results_frame["Predictions"].to_list()

    cnf_matrix = confusion_matrix(real_labels, pred_labels, normalize="true")
    disp = ConfusionMatrixDisplay(cnf_matrix)
    fig = plt.figure(figsize=(5, 5))
    # sns.heatmap(cnf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=materials, yticklabels=materials)
    # if not deep:
    #     disp.plot(colorbar=False)
    # else:
    #     disp.plot(colorbar=True)
    disp.plot(colorbar=True)    # Maintain same size of figure

    plt.yticks(np.arange(len(materials)), materials)
    plt.xticks(
        np.arange(len(materials)),
        materials,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    # plt.imshow(cnf_matrix, vmin=0, vmax=100, cmap="viridis")
    plt.title("Control System SNN Confusion matrix")
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.savefig(f"{output_path}/training_confusion.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
