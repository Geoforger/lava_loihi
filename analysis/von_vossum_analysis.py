import pandas as pd
import seaborn as sb
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

plt.rcParams.update(
    {
        "figure.figsize": (15, 10),  # (3.5,2.5),
        "font.size": 16,  # Set the font size
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
    speeds = np.arange(15,65,10)
    results_path = "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/tex_tex_speed_similarity_data.npy"
    output_path = "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/"
    results = np.load(results_path)
    # print(results)

    results_frame_path = "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/averaged_tex_tex_distance_speed_55.csv"
    results_frame = pd.read_csv(results_frame_path)
    textures = list(results_frame)
    textures.pop(0)
    print(textures)
    print(speeds)

    # results = results.drop(results.columns[0], axis=1)
    # textures = list(results)

    # sb.heatmap(data=results, annot=True, cmap="viridis")

    # plt.xticks(
    #     np.arange(len(textures)) + 0.5,
    #     textures,
    #     rotation=0,
    #     ha="center",
    #     rotation_mode="anchor",
    # )
    # # Centering the y-axis ticks, setting them horizontal, and moving them left
    # plt.yticks(
    #     np.arange(len(textures)) + 0.5,
    #     textures,
    #     rotation=0,
    #     ha="right",
    #     va="center",
    #     rotation_mode="anchor"
    # )

    # # Adjusting label position using label.set_position
    # for label in plt.gca().get_yticklabels():
    #     label.set_horizontalalignment('right')
    #     label.set_position((-0.01, label.get_position()[1]))  # Adjust the -0.3 value as needed

    # plt.title("Average Texture Van Rossum Distance Matrix During Acceleration")
    # plt.tight_layout()
    # plt.savefig(f"{output_path}/starting_van_rossum_matrix_{speed}.png", dpi=300, bbox_inches="tight")
    # plt.show()
    for idx, speed in enumerate(speeds):

        # MDS Analysis
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        slice = results[:, :, idx]
        mds_coords = mds.fit_transform(slice)

        # Plotting the MDS result
        plt.figure(figsize=(10, 8))
        plt.scatter(mds_coords[:, 0], mds_coords[:, 1], s=100)

        # Annotating the points with texture names
        for i, texture in enumerate(textures):
            plt.annotate(
                texture,
                (mds_coords[i, 0], mds_coords[i, 1]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="center",
            )

        plt.title(
            f"Multidimensional Scaling of Average Van Rossum Distances During Acceleration at Speed {speed}mm/s"
        )
        plt.xlabel("MDS Dimension 1")
        plt.ylabel("MDS Dimension 2")
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.grid(True)
        plt.savefig(
            f"{output_path}/starting_van_rossum_mds_{speed}.png", dpi=300, bbox_inches="tight"
        )
        # plt.show()
        plt.close()

if __name__ == "__main__":
    main()
