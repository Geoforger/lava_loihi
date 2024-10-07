import torch
import glob
import numpy as np
from torch.utils.data import Dataset
import lava.lib.dl.slayer as slayer
import os
import sys
import pandas as pd
from ast import literal_eval
import re

sys.path.append("..")
from utils.utils import nums_from_string
from utils.data_processor import DataProcessor


class ForceSpeedDataset(Dataset):
    def __init__(
        self,
        path,
        train,
        valid=False,
        texture=True,
        sampling_time=10
    ) -> None:

        super(ForceSpeedDataset, self).__init__()

        # Load in meta file and params from preprocessing step
        meta = pd.read_csv(f"{path}/meta.csv")
        self.sample_length = meta["cuttoff"].iloc[0]
        # self.x_size = meta["X Size"].iloc[0]
        # self.y_size = meta["Y Size"].iloc[0]
        meta['output_shape'] = meta['output_shape'].apply(literal_eval)
        self.x_size, self.y_size = meta["output_shape"].iloc[0]
        self.sampling_time = sampling_time
        self.num_time_bins = int(self.sample_length / self.sampling_time)
        self.train = train
        self.texture = texture

        # Set path to whichever dataset you want
        if self.train is True:
            self.PATH = f"{path}/train/"
        else:
            self.PATH = f"{path}/test/"

        if valid is True:
            self.PATH = f"{path}/valid/"

        self.samples = glob.glob(f"{self.PATH}*_on.pickle.npy")

    # Function to retrieve spike data from index
    def __getitem__(self, index):
        filename = self.samples[index]
        force = nums_from_string(filename)[-4]
        speed = nums_from_string(filename)[-3]
        label = nums_from_string(filename)[-2]

        event = DataProcessor.load_data_np(filename)
        event.create_events()

        event = event.data

        spike = torch.from_numpy(event.to_tensor(
            sampling_time=1, dim=(1, self.y_size, self.x_size, self.num_time_bins)
        )).float()

        return spike.reshape(-1, self.num_time_bins), label, speed, force, filename

    # Function to find length of dataset
    def __len__(self):
        return len(self.samples)


class ForceSpeedDatasetNetx(Dataset):
    def __init__(
        self, path, train, valid=False, texture=True, sampling_time=10
    ) -> None:

        super(ForceSpeedDatasetNetx, self).__init__()

        # Load in meta file and params from preprocessing step
        meta = pd.read_csv(f"{path}/meta.csv")
        self.sample_length = meta["cuttoff"].iloc[0]
        meta["output_shape"] = meta["output_shape"].apply(literal_eval)
        self.x_size, self.y_size = meta["output_shape"].iloc[0]
        self.sampling_time = sampling_time
        self.num_time_bins = int(self.sample_length / self.sampling_time)
        self.train = train
        self.texture = texture

        # Set path to whichever dataset you want
        if self.train is True:
            self.PATH = f"{path}/train/"
        else:
            self.PATH = f"{path}/test/"

        if valid is True:
            self.PATH = f"{path}/valid/"

        self.samples = glob.glob(f"{self.PATH}*_on.pickle.npy")

        with open("tests.log", "r") as f:
            log_addresses = set()
            for line in f:
                # Use regex to extract the address string in the new format
                match = re.search(r"INFO:root:(/.*)", line)
                if match:
                    address = match.group(1)
                    log_addresses.add(address)

        self.samples = [addr for addr in self.samples if addr not in log_addresses]
        print(len(self.samples))

    # Function to retrieve spike data from index
    def __getitem__(self, index):
        filename = self.samples[index]
        label = nums_from_string(filename)[-2]

        event = DataProcessor.load_data_np(filename)
        event.create_events()

        event = event.data

        spike = torch.from_numpy(
            event.to_tensor(
                sampling_time=1, dim=(1, self.y_size, self.x_size, self.num_time_bins)
            )
        ).float()

        return spike.reshape(-1, self.num_time_bins), label, filename

    # Function to find length of dataset
    def __len__(self):
        return len(self.samples)
