import torch
import glob
import numpy as np
from torch.utils.data import Dataset
import lava.lib.dl.slayer as slayer
import os
import sys
import pandas as pd

sys.path.append("..")
from utils.utils import nums_from_string


class AoDataset(Dataset):
    def __init__(
        self,
        path,
        train,
        valid=False,
        texture=True,
        sampling_time=1
    ) -> None:

        super(AoDataset, self).__init__()

        # Load in meta file and params from preprocessing step
        meta = pd.read_csv(f"{path}.meta.csv")
        self.sample_length = meta["Sample Length (ms)"].iloc[0]
        self.x_size = meta["X Size"].iloc[0]
        self.y_size = meta["Y Size"].iloc[0]
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

        self.samples = glob.glob(f"{self.PATH}*.npy")

    # Function to retrieve spike data from index
    def __getitem__(self, index):
        filename = self.samples[index]

        # Get the folder name that contains the file for label
        if self.texture is True:
            label = nums_from_string(filename)[-2]
        else:
            # TODO: Write code for if we want speed
            pass

        event = slayer.io.read_np_spikes(filename)
        spike = event.fill_tensor(
            torch.zeros(
                1, self.y_size, self.x_size, self.num_time_bins, requires_grad=False
            ),
            sampling_time=self.sample_length,
        )

        return spike.reshape(-1, self.num_time_bins), label

    # Function to find length of dataset
    def __len__(self):
        return len(self.samples)
