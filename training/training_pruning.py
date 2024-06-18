# Import torch & lava libraries
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader

# from lava.lib.dl.netx import hdf5
import h5py
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import time
import glob
import csv
import pandas as pd
from ast import literal_eval

# Import the data processing class and data collection class
from loihi_dataset import Dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

# Multi GPU
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


# ## Define network structure as before
class Network(torch.nn.Module):  # Define network
    def __init__(self, output_neurons, x_size, y_size, dropout=0.1, pruning=0.3):
        super(Network, self).__init__()

        neuron_params = {
            "threshold": 1.25,  # Previously 1.25 # THIS WAS THE CHANGE I MADE BEFORE GOING HOME 22/6/23
            "current_decay": 0.25,  # Preivously 0.25
            "voltage_decay": 0.03,  # Previously 0.03
            "tau_grad": 0.03,
            "scale_grad": 3,
            "requires_grad": True,
        }

        self.output_neurons = int(output_neurons)

        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=dropout),
        }

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(
                    neuron_params_drop,
                    x_size * y_size * 1,
                    100,
                    weight_norm=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop,
                    100,
                    self.output_neurons,
                    weight_norm=True,
                ),
            ]
        )

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)

        return spike

    def grad_flow(self, path):
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + "gradFlow.png")
        plt.close()

        return grad


def dataset_split(PATH, profs, textures, val_ratio, train_ratio=0.8):
    """Function to split a given directory of data into a training and test split after seperating data for validation

    Args:
        PATH (str): Path to the data directory
        profs (list): List of profiles within dataset
        textures (list): List of textures within dataset
        train_ratio (float, optional): Ratio of training to testing data. Defaults to 0.8.
    """
    filenames = glob.glob(f"{PATH}/*.npy")

    if os.path.exists(f"{PATH}/train/") and os.path.exists(f"{PATH}/test/"):
        if (
            input(
                f"Train & Test directories exist on dataset path {PATH}. Overwrite? This WILL overwrite both directories (y,N)"
            )
            != "y"
        ):
            print("Not overwriting current directories")
            return

    os.makedirs(f"{PATH}/train/", exist_ok=False)
    os.makedirs(f"{PATH}/test/", exist_ok=False)
    os.makedirs(f"{PATH}/valid/", exist_ok=False)

    # Take a given ratio as the validation set
    validation, train_test = train_test_split(
        filenames, train_size=val_ratio, test_size=1 - train_ratio
    )

    # Create the train/test/split
    train, test = train_test_split(
        train_test, train_size=train_ratio, test_size=1 - train_ratio
    )

    # Copy them into the train/test folder
    for f in validation:
        f_s = f.split("/")[-1]
        os.system(f"cp {f} {f'{PATH}/valid/{f_s}'}")
    for f in train:
        f_s = f.split("/")[-1]
        os.system(f"cp {f} {f'{PATH}/train/{f_s}'}")
    for f in test:
        f_s = f.split("/")[-1]
        os.system(f"cp {f} {f'{PATH}/test/{f_s}'}")

    print("Split files into train test folders")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return dataloader


def cleanup():
    dist.destroy_process_group()
    print("Destroyed process groups...")


def objective(rank, world_size, OUTPUT_PATH, training_params, network_params):
    setup(rank, world_size)

    # Training params
    num_epochs = training_params["num_epochs"]
    lr_epoch = training_params["lr_epoch"]
    learning_rate = training_params["learning_rate"]
    lr_factor = training_params["lr_factor"]
    batch_size = training_params["batch_size"]
    true_rate = training_params["true_rate"]
    false_rate = training_params["false_rate"]

    # Network params
    output_neurons = network_params["output_neurons"]
    dropout = network_params["dropout"]
    pruning = network_params["pruning"]

    # Create network and distributed object
    net = Network(
        output_neurons=output_neurons,
        dropout=dropout,
        pruning=pruning
    ).to(rank)

    if rank == 0:
        print(net)

    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # TODO: 1) Play around with different optimisers
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Initialise error module
    error = slayer.loss.SpikeRate(
        true_rate=true_rate, false_rate=false_rate, reduction="sum"
    ).to(rank)

    # Initialise stats and training assistants
    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict
    )

    ##################################
    # Initialise datasets
    ##################################
    training_set = Dataset()
    testing_set = Dataset()

    train_loader = prepare(
        training_set, rank, world_size, batch_size=batch_size, num_workers=num_workers
    )
    test_loader = prepare(
        testing_set, rank, world_size, batch_size=batch_size, num_workers=num_workers
    )

    # Training loop
    tic = time.time()
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\nEpoch {epoch}")
        epoch_tic = time.time()
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        # Reduce learning rate by factor every lr_epoch epochs
        # Check to prevent lowering lr on first epoch
        if epoch != 0:
            if (epoch % lr_epoch) == 0:
                learning_rate /= lr_factor
                assistant.reduce_lr(lr_factor)
                if rank == 0:
                    print(f"Learning rate reduced to {learning_rate}")

        # Training loop
        if rank == 0:
            print("Training...")
        for _, (input_, label) in enumerate(train_loader):
            _ = assistant.train(input_, label)

        # Testing loop
        if rank == 0:
            print("Testing...")
        for _, (input_, label) in enumerate(test_loader):
            _ = assistant.test(input_, label)

        # TODO: This should be changed to look at the output stats file instead of the stats object
        if stats.testing.best_accuracy:
            torch.save(net.module.state_dict(), f"{OUTPUT_PATH}/network.pt")

        epoch_timing = (time.time() - epoch_tic) / 60
        if rank == 0:
            print(f"\rTime taken for epoch: {np.round(epoch_timing, 2)}mins")

        stats.update()
        stats.save(OUTPUT_PATH + "/")
        # net.grad_flow(OUTPUT_PATH + '/')
        print(stats)

    time.sleep(5)
    # TODO: This should be changed to look at the output stats file instead of the stats object
    # Save the best network to a hdf5 file
    net.module.load_state_dict(torch.load(f"{OUTPUT_PATH}/network.pt"))
    net.module.export_hdf5(f"{OUTPUT_PATH}/network.net")

    toc = time.time()
    train_timing = (toc - tic) / 60
    if rank == 0:
        print("Finished training")
        print(f"\rTime taken to train network: {np.round(train_timing, 2)}mins")

    # TODO: Any validation step goes here

    cleanup()
    time.sleep(3)


def main():
    training_params = {
        "num_epochs": 100,
        "lr_epoch": 25,
        "learning_rate": 0.1,
        "lr_factor": 33,
        "batch_size": 215,
        "true_rate": 0.8,
        "false_rate": 0.02
    }

    network_params = {
        "output_neurons": 12,
        "dropout": 0.3,
        "pruning": 0.3
    }

    # Run training function
    for x in y:
        FOLDER_PATH = "../networks/tests/"
        OUTPUT_PATH = os.path.join(FOLDER_PATH, f"tests/{time.time()}/")
        os.mkdir(OUTPUT_PATH)

        # Distribute training across 3 GPUs
        world_size = 3
        mp.spawn(
            objective,
            args=[world_size, OUTPUT_PATH, training_params, network_params],
            nprocs=world_size
        )


if __name__ == "__main__":
    main()
