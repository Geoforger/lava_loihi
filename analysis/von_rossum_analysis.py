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
from pymuvr import distance_matrix


def load_file(file):
    with open(file, "rb") as openfile:
        try:
            data = pickle.load(openfile)
        except EOFError:
            print(EOFError)

def construct_observations(files):
    test = load_file(f)
    observation = np.empty((len(f), test.flatten().shape[0]))
    for f in files:
        sample = load_file(f)

        flat_sample = sample.flatten()
        
        for cell in range(len(flat_sample)):
            observation[f][cell] = flat_sample[cell]

    return observation

def main():
    DATASET_PATH = (
        "/media/george/T7 Shield/Neuromorphic Data/farscope2/preproc_dataset_spatial/"
    )
    files = glob.glob(f"{DATASET_PATH}/*.npy")
    dataset_meta = pd.read_csv(f"{DATASET_PATH}/meta.csv", index_col=0).T

    speeds = np.arange(5,65,5)
    depths = np.arange(0.5, 2.0, 0.5)
    textures = dataset_meta["Textures"].iloc[0]
    textures = literal_eval(textures)
    samples = 100
    n_speeds = len(speeds)
    n_depths = len(depths)
    n_tex = len(textures)

    tex = [f for f in files if f.split("-")[-2] == 2]
    construct_observations(tex)

    # Every sample is compared against every non-same texture sample
    #                                textures-1 * speeds * depths * samples
    # Every sample is compared against (12 * 9 * (100 * 3))
    # distance_data = np.zeros((n_tex, n_speeds, samples * n_depths), dtype=np.float32)

    # possible_pairs = itertools.combinations(textures, 2)

    # print("Starting process pool...")
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = [executor.submit(von_rossum_distance_binary, **arg_dicts[file]) for file in range(len(filenames))]

    #     print("Getting results...")
    #     for f in concurrent.futures.as_completed(results):
    #         _, shape = f.result()


if __name__ == "__main__":
    main()
