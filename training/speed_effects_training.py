# Import torch & lava libraries
import lava.lib.dl.slayer as slayer
import torch
from torch.utils.data import DataLoader

# from lava.lib.dl.netx import hdf5
import h5py
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.metrics import accuracy_score

# Import the data processing class and data collection class
from force_speed_dataset import ForceSpeedDataset
import sys
sys.path.append("..")
from utils.utils import dataset_split

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
            "threshold": 1.15,  # Previously 1.25
            "current_decay": 0.25,  # Preivously 0.25
            "voltage_decay": 0.03,  # Previously 0.03
            "tau_grad": 0.03,
            "scale_grad": 3,
            "requires_grad": True,
        }

        self.output_neurons = int(output_neurons)
        self.hidden_layer = 350

        neuron_params_drop = {
            **neuron_params,
            "dropout": slayer.neuron.Dropout(p=dropout),
        }

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.cuba.Dense(
                    neuron_params_drop,
                    x_size * y_size * 1,
                    self.hidden_layer,
                    weight_norm=True,
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop,
                    self.hidden_layer,
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
    
    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


def setup(rank, world_size):  
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader


def cleanup():
    dist.destroy_process_group()
    print("Destroyed process groups...")


def objective(rank, world_size, DATASET_PATH, true_rate):
    setup(rank, world_size)

    #############################
    # Output params
    #############################
    start_time = int(time.time())
    OUTPUT_PATH = f"/media/george/T7 Shield/Neuromorphic Data/George/arm_networks/arm_test_nonorm_{start_time}/"
    if rank == 0:
        os.makedirs(OUTPUT_PATH, exist_ok=False)
        # os.makedirs(OUTPUT_PATH+"spikes/", exist_ok=False)

    # Read meta for x, y sizes of preproc data
    meta = pd.read_csv(f"{DATASET_PATH}/meta.csv")
    meta['output_shape'] = meta['output_shape'].apply(literal_eval)
    x_size, y_size = meta["output_shape"].iloc[0]

    #############################
    # Training params
    #############################
    num_epochs = 25
    learning_rate = 0.001   # Starting learning rate
    batch_size = 350
    hidden_layer = 125
    factor = 3.33   # Factor by which to divide the learning rate
    lr_epoch = 200    # Lower the learning rate by factor every lr_epoch epochs
    sample_length = 1000
    # truer_rate = 0.5

    # Initialise network and slayer assistant
    net = Network(
        output_neurons=10,
        x_size=x_size,
        y_size=y_size,
        dropout=0.3
    ).to(rank)
    if rank == 0:
        print(net)

    # Create network and distributed object
    net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # TODO: 1) Play around with different optimisers
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Initialise error module
    # TODO: 1) Play around with different error rates, etc.
    error = slayer.loss.SpikeRate(
        true_rate=true_rate, false_rate=0.02, reduction='sum').to(rank)

    # Initialise stats and training assistants
    stats = slayer.utils.LearningStats()
    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

    # Load in datasets
    training_set = ForceSpeedDataset(
        DATASET_PATH,
        train=True,
    )
    testing_set = ForceSpeedDataset(
        DATASET_PATH,
        train=False,
    )

    train_loader = prepare(training_set, rank, world_size, batch_size=batch_size, num_workers=0)
    test_loader = prepare(testing_set, rank, world_size, batch_size=batch_size, num_workers=0)

    testing_labels = []
    testing_preds = []
    speed_labels = []
    depth_labels = []

    ##################################
    # Training loop
    ##################################
    # Loop through each training epoch
    if rank == 0:
        print("Starting training loop")
    for epoch in range(num_epochs):
        tic = time.time()
        if rank == 0:
            print(f"\nEpoch {epoch}")
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        dist.barrier()

        # Reduce learning rate by factor every lr_epoch epochs
        # Check to prevent lowering lr on first epoch
        if epoch != 0:
            if (epoch % lr_epoch) == 0:
                learning_rate /= factor
                assistant.reduce_lr(factor)
                print(f"Learning rate reduced to {learning_rate}")

        # Training loop
        if rank == 0:
            print("Training...")
        for _, (input, label, _s, _f) in enumerate(train_loader):
            output = assistant.train(input, label)

        # Testing loop
        if rank == 0:
            print("Testing...")
        for _, (input, label, speed, force) in enumerate(test_loader):
            output = assistant.test(input, label)
        
        dist.barrier()  # Wait for all ranks to finish epoch
        
        if stats.testing.best_accuracy:
            if rank == 0:
                torch.save(net.module.state_dict(), f"{OUTPUT_PATH}/network.pt")

        stats.update()
        stats.save(OUTPUT_PATH + "/")

        toc = time.time()
        epoch_timing = (toc - tic) / 60
        if rank == 0:
            print(stats)
            print(f"\r[Epoch {epoch:2d}/{num_epochs}] {stats}", end="")
            print(f"\r[Epoch {epoch:2d}/{num_epochs}] {stats}", end="")
            print(f"\nTime taken for this epoch = {epoch_timing} mins")

    # TODO: This should be changed to look at the output stats file instead of the stats object
    # Save the best network to a hdf5 file
    if rank == 0:
        net.module.load_state_dict(torch.load(f"{OUTPUT_PATH}/network.pt"))
        net.module.export_hdf5(f"{OUTPUT_PATH}/network.net")
    print("Finished training")
    
    # Perform validation loop
    validation_set = ForceSpeedDataset(
        DATASET_PATH,
        train=False,
        valid=True
    )

    val_loader = prepare(validation_set, rank, world_size, batch_size=batch_size, num_workers=0)
    
    if rank == 0:
        print("Performing validation...")
    for _, (input, label, speed, force) in enumerate(val_loader):
        output = assistant.test(input, label)
        for l in range(len(slayer.classifier.Rate.predict(output))):
            testing_labels.append(int(label[l].cpu()))
            testing_preds.append(int(
                slayer.classifier.Rate.predict(output)[l].cpu()
            ))
            speed_labels.append(int(speed[l]))
            depth_labels.append(float(force[l]))

            # Save output spikes
            # np.save(f"{OUTPUT_PATH}/spikes/{filename[l]}", output[l].cpu())
    print(f"Validation accuracy: {accuracy_score(testing_labels, testing_preds)}")

    # Save output stats for testing
    test_stats = pd.DataFrame(data={
        "Labels": testing_labels,
        "Predictions": testing_preds,
        "Speeds": speed_labels,
        "Depths": depth_labels,
        "True Rate": true_rate
    })
    test_stats.to_csv(f"{OUTPUT_PATH}/output_labels.csv")
    print(f"Accuracy of test {start_time}: {stats.testing.best_accuracy}")

    cleanup()


def main():
    DATASET_PATH = "/media/george/T7 Shield/Neuromorphic Data/George/speed_depth_preproc_downsampled/"
    train_ratio = 0.6
    valid_ratio = 0.2
    
    # Train test split
    print(f"Splitting dataset into train/test with ratio: {train_ratio}")
    dataset_split(DATASET_PATH, train_ratio=train_ratio, valid_ratio=valid_ratio)
    
    world_size = 3
    
    for r in np.arange(0.7, 1.0, 0.1):
        for _ in range(25):
            mp.spawn(
                objective,
                args=[world_size, DATASET_PATH, r],
                nprocs=world_size
            )
            time.sleep(3)

if __name__ == "__main__":
    main()
