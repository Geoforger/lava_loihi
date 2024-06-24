import torch
import glob
import numpy as np
from torch.utils.data import Dataset
import lava.lib.dl.slayer as slayer
import os
import sys
sys.path.append("..")
from utils.utils import nums_from_string

class DavisDataset(Dataset):
    def __init__(
        self,
        path,
        train,
        valid=False,
        x_size=240,
        y_size=180,
        sampling_time=1,
        sample_length=1000,
    ) -> None:

        super(DavisDataset, self).__init__()

        self.train = train

        # Set path to whichever dataset you want
        if self.train is True:
            self.PATH = f"{path}/train/"
        else:
            self.PATH = f"{path}/test/"

        if valid is True:
            self.PATH = f"{path}/valid/"

        self.samples = glob.glob(f"{self.PATH}*.npy")
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length / sampling_time)
        self.x_size = x_size
        self.y_size = y_size

    # Function to retrieve spike data from index
    def __getitem__(self, index):
        filename = self.samples[index]

        # Get the folder name that contains the file for label
        label = nums_from_string(filename)[-1]

        event = slayer.io.read_np_spikes(filename)
        spike = event.fill_tensor(
            torch.zeros(
                1, self.y_size, self.x_size, self.num_time_bins, requires_grad=False
            ),
            sampling_time=self.sampling_time,
        )

        return spike.reshape(-1, self.num_time_bins), label

    # Function to find length of dataset
    def __len__(self):
        return len(self.samples)
