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
    try:
        distances = pymuvr.dissimilarity_matrix(obv_a, obv_b, cos, tau, "distance")
    except Exception as e:
        print(f"Task generated an exception: {e}")
        
    mean_distance = np.mean(distances)
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

    # Construct matrix of all observations
    observation_path = Path(f"{OUTPUT_PATH}/observations.npy")

    # if not observation_path.exists():
    #     print("Constructing observation matrix...")
    #     observations = np.empty(n_tex, dtype=object)
    #     tic = time.time()
    #     for t in range(n_tex):
    #         print(f"Constructing observation for texture: {textures[t]}")
    #         tex_files = [f for f in files if f.split("-")[-2] == str(t)]
    #         observations[t] = construct_observations(tex_files)
    #     toc = time.time()
    #     print("Observation matrix constructed")
    #     print(f"Time taken to create observations: {(toc-tic)/60}mins")
        
    #     np.save(f"{OUTPUT_PATH}/observations.npy",observations, allow_pickle=True)
    #     # observations.tofile(f"{OUTPUT_PATH}/observations_txt.npy") # NOTE: This did not work
    #     print("Saved data")
    # else:
    #     observations = np.load(f"{OUTPUT_PATH}/observations.npy", allow_pickle=True)
    #     print("Imported data")
        
    # Create array to store averages of each distance matrix
    simularity_data = np.empty((n_tex, n_tex))
    self_simularity_data = np.empty(n_tex)

    # Create iterator to iterate through each texture pair
    pairs = list(itertools.combinations(np.arange(n_tex), 2))    
    print("Starting pairwise analysis...")
    
    tic = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for tex in range(n_tex):
            print(f"Analysing texture: {textures[tex]}")
            tex_files = [f for f in files if f.split("-")[-2] == str(tex)]
            tex_observations = construct_observations(tex_files)
            print("Constructed observations for texture...")
            num_observations = len(tex_observations)
            
            futures = []
            for sample in range(num_observations):
                # Get rid of sample we're comparing to
                observations_w_o_sample = [tex_observations[idx] for idx in range(num_observations) if idx != sample]
                
                for other_sample in observations_w_o_sample:
                    futures.append(executor.submit(compute_distance_pair, [tex_observations[sample]], [other_sample], cos, tau))
            
            print("All tasks submitted")
            total_tasks = len(futures)
            completed_tasks = 0
            
            sample_means = []       
            for future in concurrent.futures.as_completed(futures):
                try:
                    mean_distance = future.result()
                    sample_means.append(mean_distance)
                except Exception as e:
                    print(f"Task generated an exception: {e}")
            
                completed_tasks += 1
                if completed_tasks % 1500 == 0:
                    print(f"Progress: {completed_tasks}/{total_tasks} tasks completed ({(completed_tasks / total_tasks) * 100:.2f}%)")
                    
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
