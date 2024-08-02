import numpy as np
import pandas as pd
import glob
import pickle
from ast import literal_eval
import sys
import itertools
import pymuvr
import time
from pathlib import Path
from numba import cuda, float64


@cuda.jit
def compute_distance_kernel(obv_a, obv_b, cos, tau, distances):
    i, j = cuda.grid(2)
    if i < obv_a.shape[0] and j < obv_b.shape[0]:
        diff = obv_a[i] - obv_b[j]
        distances[i, j] = np.exp(-diff * diff / tau)


def compute_distance_pair_gpu(obv_a, obv_b, cos, tau):
    obv_a = np.array(obv_a).flatten().astype(np.float64)
    obv_b = np.array(obv_b).flatten().astype(np.float64)
    distances = np.zeros((obv_a.shape[0], obv_b.shape[0]), dtype=np.float64)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(obv_a.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(obv_b.shape[0] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    compute_distance_kernel[blockspergrid, threadsperblock](
        obv_a, obv_b, cos, tau, distances
    )
    cuda.synchronize()

    mean_distance = np.mean(distances)
    return mean_distance


def construct_observations(files):
    observation = []
    for f in files:
        sample = np.load(f, allow_pickle=True)
        flat_sample = sample.flatten()
        observation.append([list(l) if len(l) != 0 else [] for l in flat_sample])
    return observation


def main():
    ORIGINAL_DATASET = (
        "/media/george/T7 Shield/Neuromorphic Data/George/all_speeds_sorted/"
    )
    DATASET_PATH = "/media/george/T7 Shield/Neuromorphic Data/George/preproc_dataset_spatial_start_pickle/"
    OUTPUT_PATH = (
        "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/"
    )
    files = glob.glob(f"{DATASET_PATH}/*.npy")
    print(f"Num samples: {len(files)}")
    dataset_meta = pd.read_csv(f"{ORIGINAL_DATASET}/meta.csv", index_col=0).T
    proc_meta = pd.read_csv(f"{DATASET_PATH}/meta.csv")

    speeds = np.arange(5, 65, 5)
    depths = np.arange(0.5, 2.0, 0.5)
    textures = dataset_meta["Textures"].iloc[0]
    textures = literal_eval(textures)
    samples = 100
    n_speeds = len(speeds)
    n_depths = len(depths)
    n_tex = len(textures)
    print(f"Num textures: {n_tex}")
    proc_meta["output_shape"] = proc_meta["output_shape"].apply(literal_eval)
    cos = 0.1
    tau = 1.0
    BATCH_SIZE = 1

    simularity_data = np.empty((n_tex, n_tex))
    self_simularity_data = np.empty(n_tex)

    pairs = list(itertools.combinations(np.arange(n_tex), 2))
    print("Starting pairwise analysis...")

    tic = time.time()

    for tex in range(n_tex):
        print(f"Analysing texture: {textures[tex]}")
        tex_files = [f for f in files if f.split("-")[-2] == str(tex)]
        tex_observations = construct_observations(tex_files)
        print("Constructed observations for texture...")
        num_observations = len(tex_observations)

        sample_means = []
        for sample in range(num_observations):
            observations_w_o_sample = [
                tex_observations[idx]
                for idx in range(num_observations)
                if idx != sample
            ]

            for other_sample in observations_w_o_sample:
                try:
                    mean_distance = compute_distance_pair_gpu(
                        [tex_observations[sample]], [other_sample], cos, tau
                    )
                    sample_means.append(mean_distance)
                except Exception as e:
                    print(f"Task generated an exception: {e}")

        texture_mean = np.mean(sample_means)
        self_simularity_data[tex] = texture_mean
        print(f"Average distance within texture {textures[tex]} data: {texture_mean}")

    toc = time.time()
    print(f"Total time taken: {(toc-tic)/60}mins")
    print(self_simularity_data)
    np.save(f"{OUTPUT_PATH}/self_simularity_data.npy", self_simularity_data)

    data_frame = pd.DataFrame(data=self_simularity_data, columns=textures)
    print(data_frame)
    data_frame.to_csv(f"{OUTPUT_PATH}/averaged_self_distance.csv")


if __name__ == "__main__":
    main()
