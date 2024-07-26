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
from pathlib import Path


def construct_observations(files):
    observation = []
    # Load each sample into the overall observation for this texture
    for f in files:
        sample = np.load(f, allow_pickle=True)
        flat_sample = sample.flatten()
        observation.append([list(l) if len(l)!=0 else [] for l in flat_sample])
        
    return observation

def compute_distance_pair(obv_a, obv_b, cos, tau):
    mean_distance = np.mean(pymuvr.dissimilarity_matrix(obv_a, obv_b, cos, tau, "distance"))
    return mean_distance


def main():
    ORIGINAL_DATASET = "/media/george/T7 Shield/Neuromorphic Data/George/all_speeds_sorted/"
    DATASET_PATH = "/media/george/T7 Shield/Neuromorphic Data/George/preproc_dataset_spatial_start_pickle/"
    OUTPUT_PATH = "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/"
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
    BATCH_SIZE = 1
        
    # Create array to store averages of each distance matrix
    simularity_data = np.zeros((n_tex, n_tex))

    # Create iterator to iterate through each texture pair
    pairs = list(itertools.combinations(np.arange(n_tex), 2))
    print("Starting pairwise analysis...")
    
    tic = time.time()
    # NOTE: This does not perform self comparisons
    for (tex_1, tex_2) in pairs:
        print(f"Analysing textures: {textures[tex_1]} and {textures[tex_2]}")
        texture_files = [f for f in files if f.split("-")[-2] == str(tex_1) or f.split("-")[-2] == str(tex_2)]
        tex_files_1 = [f for f in texture_files if f.split("-")[-2] == str(tex_1)]
        tex_files_2 = [f for f in texture_files if f.split("-")[-2] == str(tex_2)]
        tex_observations_1 = construct_observations(tex_files_1)
        tex_observations_2 = construct_observations(tex_files_2)
        print("Constructed observations for textures...")
        
        num_tex1_observations = len(tex_observations_1)
                
        for t1_sample in range(num_tex1_observations):
            for t2_sample in tex_observations_2:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(compute_distance_pair, [tex_observations_1[t1_sample]], [t2_sample], cos, tau)]
        
        sample_means = []       
        for future in concurrent.futures.as_completed(futures):
            mean_distance = future.result()
            sample_means.append(mean_distance)

        pair_mean = np.mean(sample_means)
        print(f"Average distance between textures: {textures[tex_1]} and {textures[tex_2]}: {pair_mean}")
        simularity_data[tex_1, tex_2], simularity_data[tex_2, tex_1] = pair_mean

    toc = time.time()
    print(f"Total time taken: {(toc-tic)/60}mins")
    print(simularity_data)
    np.save(f"{OUTPUT_PATH}/simulariy.npy", simularity_data)
    
    data_frame = pd.DataFrame(data=simularity_data, columns=textures, index=textures)
    print(data_frame)
    data_frame.to_csv(f"{OUTPUT_PATH}/averaged_distance.csv")

if __name__ == "__main__":
    main()
