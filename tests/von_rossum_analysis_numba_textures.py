import numpy as np
import pandas as pd
import glob
from ast import literal_eval
import itertools
import time
from numba import cuda, float64
import math

@cuda.jit
def compute_distance_kernel(obv_a, obv_b, cos, tau, distances, size_a, size_b):
    i, j = cuda.grid(2)
    if i < size_a and j < size_b:
        diff = obv_a[i] - obv_b[j]
        distances[i * size_b + j] = math.exp(-diff * diff / tau)


def compute_distance_pair_gpu(obv_a, batch_b, cos, tau):
    flat_obv_a = np.array(obv_a).flatten().astype(np.float64)
    flat_obv_b = np.concatenate([np.array(obv).flatten().astype(np.float64) for obv in batch_b])
    
    size_a = flat_obv_a.shape[0]
    size_b = flat_obv_b.shape[0] // len(batch_b)  # Each batch element should have the same size

    distances = np.zeros(size_a * size_b, dtype=np.float64)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(size_a / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(size_b / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    compute_distance_kernel[blockspergrid, threadsperblock](flat_obv_a, flat_obv_b, cos, tau, distances, size_a, size_b)
    cuda.synchronize()

    distances = distances.reshape(size_a, size_b)
    mean_distances = np.mean(distances, axis=1)
    return mean_distances


def construct_observations(files):
    observations = []
    for f in files:
        sample = np.load(f, allow_pickle=True)
        flat_sample = sample.flatten()
        flat_sample = np.array([item for sublist in flat_sample for item in sublist], dtype=np.float64)
        observations.append(flat_sample)
    return observations


def main():
    ORIGINAL_DATASET = (
        "/media/farscope2/T7 Shield/Neuromorphic Data/George/all_speeds_sorted/"
    )
    DATASET_PATH = "/media/farscope2/T7 Shield/Neuromorphic Data/George/preproc_dataset_spatial_start_pickle/"
    OUTPUT_PATH = (
        "/media/farscope2/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/"
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
    BATCH_SIZE = 1000
    MAX_BATCH_SIZE = 4000

    similarity_data = np.zeros((n_tex, n_tex))

    print("Starting pairwise analysis...")

    tic = time.time()

    sample_counter = 0
    total_comparisons = (
        sum(len(glob.glob(f"{DATASET_PATH}/*-{tex}-*.npy")) for tex in range(n_tex))
        ** 2
        / 2
    )
    pairs = list(itertools.combinations(np.arange(n_tex), 2))

    for tex_a,tex_b in pairs:
        print(f"Analysing texture: {textures[tex_a]}")
        tex_files_a = [f for f in files if f.split("-")[-2] == str(tex_a)]
        tex_observations_a = construct_observations(tex_files_a)
        num_observations_a = len(tex_observations_a)

        tex_files_b = [f for f in files if f.split("-")[-2] == str(tex_b)]
        tex_observations_b = construct_observations(tex_files_b)
        num_observations_b = len(tex_observations_b)

        sample_means = []

        for sample_idx_a in range(num_observations_a):
            obv_a = tex_observations_a[sample_idx_a]

            for batch_start in range(0, num_observations_b, BATCH_SIZE):
                batch_b = [
                    tex_observations_b[idx]
                    for idx in range(
                        batch_start,
                        min(batch_start + BATCH_SIZE, num_observations_b),
                    )
                ]

                if len(batch_b) > 0:
                    try:
                        distances = compute_distance_pair_gpu(
                            obv_a, batch_b, cos, tau
                        )
                        sample_means.extend(distances)
                    except Exception as e:
                        print(f"Task generated an exception: {e}")

            sample_counter += 1
            percent_complete = (sample_counter / total_comparisons) * 100
            if sample_counter % 10 == 0:  # Print progress every 10 samples
                print(
                    f"Processed {sample_counter} samples ({percent_complete:.2f}% complete) for texture pair ({textures[tex_a]}, {textures[tex_b]})"
                )

        texture_mean = np.mean(sample_means)
        similarity_data[tex_a, tex_b] = texture_mean
        similarity_data[tex_b, tex_a] = texture_mean
        print(
            f"Average distance between textures {textures[tex_a]} and {textures[tex_b]}: {texture_mean}"
        )

    toc = time.time()
    print(f"Total time taken: {(toc-tic)/60}mins")
    print(similarity_data)
    np.save(f"{OUTPUT_PATH}/tex_tex_similarity_data.npy", similarity_data)

    data_frame = pd.DataFrame(data=similarity_data, columns=textures, index=textures)
    print(data_frame)
    data_frame.to_csv(f"{OUTPUT_PATH}/averaged_tex_tex_distance.csv")


if __name__ == "__main__":
    main()
