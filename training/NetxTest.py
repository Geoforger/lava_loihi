import numpy as np
import logging
from lava.lib.dl import netx
import lava.lib.dl.slayer as slayer
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io.source import RingBuffer as InBuffer
from lava.proc.io.sink import RingBuffer as OutBuffer
import torch
from torch.utils.data import DataLoader
import time
import pandas as pd

from force_speed_dataset import ForceSpeedDatasetNetx as ForceSpeedDataset

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


def test_with_netx(net_path, inp, sample_length=1000):
    net = netx.hdf5.Network(
        net_config=net_path
    )
    run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
    run_condition = RunSteps(num_steps=sample_length)

    input_data = inp.squeeze()

    input_buffer = InBuffer(data=input_data.numpy())
    output_buffer = OutBuffer(shape=net.out.shape, buffer=sample_length)

    input_buffer.s_out.connect(net.inp)
    net.out.connect(output_buffer.a_in)

    # with net:
    net.run(condition=run_condition, run_cfg=run_cfg)
    out = output_buffer.data.get()
    net.stop()

    output_totals = np.sum(out, axis=1)
    decision = np.argmax(output_totals)

    return decision


def test_with_torch(net_path, x_size, y_size, inp, sample_length=1000):
    # Load network
    net = Network(10, x_size, y_size)
    net.load_state_dict(torch.load(net_path, map_location=torch.device("cpu")))
    net.eval()

    with torch.no_grad():
        output = net(inp.unsqueeze(0))

    output = output.numpy()
    total_spikes = np.sum(output.squeeze(0), axis=1)
    decision = np.argmax(total_spikes, axis=0)

    return decision


def main():
    logging.basicConfig(filename="tests.log", level=logging.INFO)

    dataset_path="/media/farscope2/T7 Shield/Neuromorphic Data/George/speed_depth_preproc_downsampled/"

    # Load testing set
    testing_set = ForceSpeedDataset(
        dataset_path,
        train=False,
    )

    # for idx in range(len(testing_set)):
    input, label, filename = testing_set.__getitem__(0)

    netx_pred = test_with_netx(
        net_path="/home/farscope2/Documents/PhD/lava_loihi/networks/best_arm_network/network.net",
        # dataset_path=dataset_path,
        inp=input,
        sample_length=1000
    )

    torch_pred = test_with_torch(
        net_path="/home/farscope2/Documents/PhD/lava_loihi/networks/best_arm_network/network.pt",
        x_size=67,
        y_size=69,
        inp=input,
        sample_length=1000,
    )

    print(f"Label: {label} \nTorch: {torch_pred} \nNetx: {netx_pred} \n\n")
    logging.info(filename)

    output_dict = {
        "Label": label,
        "Torch": torch_pred,
        "Netx": netx_pred
    }

    output_frame = pd.DataFrame(
        data=output_dict, columns=list(output_dict.keys()), index=[0]
    )
    output_frame.to_csv("test_outputs.csv", mode="a", header=False)


if __name__ == "__main__":
    main()
