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
from loihi_dataset import DavisDataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

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


def main():
    #############################
    # Output params
    #############################
    OUTPUT_PATH = "/home/farscope2/Documents/PhD/lava_loihi/networks/"
    DATASET_PATH = "/home/farscope2/Documents/PhD/lava_loihi/data/datasets/"

    #############################
    # Training params
    #############################
    num_epochs = 5
    learning_rate = 0.0001   # Starting learning rate
    batch_size = 25
    hidden_layer = 125
    factor = 3.33   # Factor by which to divide the learning rate
    lr_epoch = 4    # Lower the learning rate by factor every lr_epoch epochs
    sample_length = 1000
    # truer_rate = 0.5

    testing_labels = []
    testing_preds = []

    # Initialise network and slayer assistant
    net = Network(hidden_size=hidden_layer)

    # TODO: 1) Play around with different optimisers
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Initialise error module
    # TODO: 1) Play around with different error rates, etc.
    error = slayer.loss.SpikeRate(
        true_rate=0.5, false_rate=0.02, reduction='sum')

    # Initialise stats and training assistants
    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

    # Get all file names into kernal
    files = glob.glob(f"{DATASET_PATH}/*.pickle")

    # Load in datasets
    training_set = DavisDataset(
        DATASET_PATH,
        train=True,
        x_size=x_size,
        y_size=y_size,
        sample_length=sampling_length,
    )
    testing_set = DavisDataset(
        dataset_path,
        train=False,
        x_size=x_size,
        y_size=y_size,
        sample_length=sampling_length,
    )

    train_loader = DataLoader(
        dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        dataset=testing_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    ##################################
    # Training loop
    ##################################
    # Loop through each training epoch
    print("Starting training loop")
    for epoch in range(num_epochs):
        tic = time.time()
        print(f"\nEpoch {epoch}")

        # Reduce learning rate by factor every lr_epoch epochs
        # Check to prevent lowering lr on first epoch
        if epoch != 0:
            if (epoch % lr_epoch) == 0:
                learning_rate /= factor
                assistant.reduce_lr(factor)
                print(f"Learning rate reduced to {learning_rate}")

        # Training loop
        for i, (input_, label) in enumerate(train_loader):
            output = assistant.train(input_, label)

        # Testing loop
        for i, (input_, label) in enumerate(test_loader):  # testing loop
            output = assistant.test(input_, label)
            if epoch == num_epochs - 1:
                for l in range(len(slayer.classifier.Rate.predict(output))):
                    testing_labels.append(label[l].cpu())
                    testing_preds.append(
                        slayer.classifier.Rate.predict(output)[l].cpu()
                    )

        if stats.testing.best_accuracy:
            torch.save(net.state_dict(), f"{OUTPUT_PATH}/network.pt")
            net.export_hdf5(f"{OUTPUT_PATH}/network.net")

    stats.update()
    stats.save(OUTPUT_PATH + "/")
    net.grad_flow(OUTPUT_PATH + "/")
    print(stats)

    toc = time.time()
    epoch_timing = (toc - tic) / 60
    print(f"\r[Epoch {epoch:2d}/{num_epochs}] {stats}", end="")
    print(f"\r[Epoch {epoch:2d}/{num_epochs}] {stats}", end="")
    print(f"\nTime taken for this epoch = {epoch_timing} mins")
    # stats.plot(figsize=(15, 5))

    print("Finished training")
    # stats.plot(figsize=(15, 5))


if __name__ == "__main__":
    main()
