import numpy as np
import pandas as pd
import glob
import pickle
from ast import literal_eval
import sys
sys.path.append("..")
from lava_loihi.utils.utils import nums_from_string
import numpy as np
import concurrent.futures
import itertools
import pymuvr


# def load_file(file):
#     with open(file, "rb") as openfile:
#         try:
#             data = pickle.load(openfile)
#         except EOFError:
#             print(EOFError)

#     return data

def construct_observations(files, shape):
    observation = np.empty((len(files), shape), dtype=object)
    # Load each sample into the overall observation for this texture
    for f in files:
        # sample = load_file(f)
        sample = np.load(f, allow_pickle=True)

        flat_sample = sample.flatten()

        for cell in range(len(flat_sample)):
            observation[f][cell] = flat_sample[cell]

    return observation

def main():
    ORIGINAL_DATASET = ("/media/farscope2/T7 Shield/Neuromorphic Data/George/all_speeds_sorted/")
    DATASET_PATH = "/media/farscope2/T7 Shield/Neuromorphic Data/George/preproc_dataset_spatial_start_pickle/"
    OUTPUT_PATH = "/home/farscope2/Documents/PhD/lava_loihi/data/dataset_analysis/"
    files = glob.glob(f"{DATASET_PATH}/*.npy")
    print(f"Num samples: {len(files)}")
    dataset_meta = pd.read_csv(f"{ORIGINAL_DATASET}/meta.csv", index_col=0).T
    proc_meta = pd.read_csv(f"{DATASET_PATH}/meta.csv")

    speeds = np.arange(5,65,5)
    depths = np.arange(0.5, 2.0, 0.5)
    textures = dataset_meta["Textures"].iloc[0]
    textures = literal_eval(textures)
    samples = 100
    n_speeds = len(speeds)
    n_depths = len(depths)
    n_tex = len(textures)
    proc_meta["output_shape"] = proc_meta["output_shape"].apply(literal_eval)
    x_size, y_size = proc_meta["output_shape"].iloc[0]
    shape = x_size * y_size

    tex_1 = [f for f in files if f.split("-")[-2] == "1"]
    spike_train_1 = construct_observations(tex_1, shape)
    tex_2 = [f for f in files if f.split("-")[-2] == "2"]
    spike_train_2 = construct_observations(tex_2, shape)

    diss_matrix = pymuvr.dissimilarity_matrix(spike_train_1, spike_train_2, 0, 0, "distance")

    print(diss_matrix)
    print(diss_matrix.shape)

    # Construct matrix of all observations
    observations = np.empty(n_tex)
    for t in range(n_tex):
        tex_files = [f for f in files if f.split("-")[-2] == t]
        observations[t] = construct_observations(tex_files, shape)

    # Create array to store averages of each distance matrix
    simularity_data = np.empty((n_tex, n_tex))

    # Create iterator to iterate through each texture pair
    pairs = list(itertools.combinations(np.arange(n_tex), 2))    

    for (t_1, t_2) in pairs:
        obv_a = observations[t_1]
        obv_b = observations[t_2]

        pair_distance = pymuvr.dissimilarity_matrix(
            obv_a, obv_b, cos=0, tau=0, mode="distance"
        )

        simularity_data[t_1, t_2], simularity_data[t_2, t_1] = np.mean(pair_distance)

    print(simularity_data)

    data_frame = pd.DataFrame(data=simularity_data, index=textures, columns=textures)
    print(data_frame)
    data_frame.to_csv(f"{OUTPUT_PATH}/averaged_distance.csv")

if __name__ == "__main__":
    main()
