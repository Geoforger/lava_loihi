import torch
import glob
from torch.utils.data import Dataset
import sys
import pandas as pd
from ast import literal_eval
import random
import time

sys.path.append("..")
from utils.utils import nums_from_string
from utils.data_processor import DataProcessor


class SimulatorDataset(Dataset):
    def __init__(
        self, path, label, sampling_time=1
    ) -> None:

        super(SimulatorDataset, self).__init__()

        # Load in meta file and params from preprocessing step
        meta = pd.read_csv(f"{path}/meta.csv")
        self.sample_length = meta["cuttoff"].iloc[0]
        meta["output_shape"] = meta["output_shape"].apply(literal_eval)
        self.x_size, self.y_size = meta["output_shape"].iloc[0]
        self.sampling_time = sampling_time
        self.num_time_bins = int(self.sample_length / self.sampling_time)

        self.PATH = f"{path}/valid/"

        self.samples = glob.glob(f"{self.PATH}*_on.pickle.npy")
        # {force} - {speed} - {label_idx} - {trial_idx}
        self.samples = [s for s in self.samples if nums_from_string(s)[-2] == label]

    # Function to retrieve spike data from index
    def __getitem__(self, speed):
        valid_samples = [s for s in self.samples if nums_from_string(s)[-3] == speed]
        # {force} - {speed} - {label_idx} - {trial_idx}
        random.seed(time.time())
        if not valid_samples:
            raise ValueError(f"No valid samples found for speed: {speed}")
        filename = random.choice(valid_samples) # Grab random sample from the dataset

        event = DataProcessor.load_data_np(filename)
        event.create_events()

        event = event.data

        spike = torch.from_numpy(
            event.to_tensor(
                sampling_time=1, dim=(1, self.y_size, self.x_size, self.num_time_bins)
            )
        ).float()

        return spike.reshape(-1, self.num_time_bins), filename

    # Function to find length of dataset
    def __len__(self):
        return len(self.samples)
