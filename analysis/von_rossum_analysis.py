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
import time

def construct_observations(files):
    observation = []
    # Load each sample into the overall observation for this texture
    for f in files:
        sample = np.load(f, allow_pickle=True)
        flat_sample = sample.flatten()
        observation.append([list(l) if l != [] else [] for l in flat_sample])

    return observation

def compute_distance_pair(obv_a, obv_b, t_1, t_2, cos, tau):
    tic = time.time()
    pair_distance = pymuvr.dissimilarity_matrix(obv_a, obv_b, cos, tau, "distance")
    toc = time.time()
    mean_distance = np.mean(pair_distance)
    print(f"Pair: Texture {t_1} and {t_2}, Time taken: {toc - tic}, Mean distance: {mean_distance}")
    return (t_1, t_2, mean_distance)

def main():
    ORIGINAL_DATASET = "/media/george/T7 Shield/Neuromorphic Data/George/all_speeds_sorted/"
    DATASET_PATH = "/media/george/T7 Shield/Neuromorphic Data/George/preproc_dataset_spatial_start_pickle/"
    OUTPUT_PATH = "/home/george/Documents/lava_loihi/data/dataset_analysis/"
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
    print(f"Num textures: {n_tex}")
    proc_meta["output_shape"] = proc_meta["output_shape"].apply(literal_eval)
    # x_size, y_size = proc_meta["output_shape"].iloc[0]
    # shape = x_size * y_size
    cos = 0.1
    tau = 1.0

    # tex_1 = [f for f in files if f.split("-")[-2] == "1"]
    # spike_train_1 = construct_observations(tex_1, shape)
    # tex_2 = [f for f in files if f.split("-")[-2] == "2"]
    # spike_train_2 = construct_observations(tex_2, shape)
    
    # # print(spike_train_1[0])
    # # print(spike_train_1.shape)
    # cos = 0.1
    # tau = 1.0

    # print("Calculating Von Rossum distances...")
    # diss_matrix = pymuvr.dissimilarity_matrix(spike_train_1, spike_train_2, cos, tau, "distance")
    # # diss_matrix = pymuvr.square_dissimilarity_matrix(spike_train_1, 0.1, 1.0, "distance")

    # print(diss_matrix)
    # print(diss_matrix.shape)

    # Construct matrix of all observations
    print("Constructing observation matrix...")
    observations = np.empty(n_tex, dtype=object)
    tic = time.time()
    for t in range(n_tex):
        print(f"Constructing observation for texture: {textures[t]}")
        tex_files = [f for f in files if f.split("-")[-2] == str(t)]
        observations[t] = construct_observations(tex_files)
    toc = time.time()
    print("Observation matrix constructed")
    print(f"Time taken to create observations: {toc-tic}")
    # Create array to store averages of each distance matrix
    simularity_data = np.empty((n_tex, n_tex))

    # Create iterator to iterate through each texture pair
    pairs = list(itertools.combinations(np.arange(n_tex), 2))    
    print("Starting pairwise analysis...")
    tic = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_distance_pair, observations[t_1], observations[t_2], t_1, t_2, cos, tau) for (t_1, t_2) in pairs]
        for future in concurrent.futures.as_completed(futures):
            t_1, t_2, mean_distance = future.result()
            simularity_data[t_1, t_2] = mean_distance
            simularity_data[t_2, t_1] = mean_distance
    toc = time.time()
    print(f"Total time taken: {toc-tic}")
    print(simularity_data)

    data_frame = pd.DataFrame(data=simularity_data, index=textures, columns=textures)
    print(data_frame)
    data_frame.to_csv(f"{OUTPUT_PATH}/averaged_distance.csv")

if __name__ == "__main__":
    main()
