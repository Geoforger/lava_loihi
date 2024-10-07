import numpy as np
import logging
from lava.lib.dl import netx
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io.source import RingBuffer as InBuffer
from lava.proc.io.sink import RingBuffer as OutBuffer
from torch.utils.data import DataLoader
import time
import pandas as pd

from force_speed_dataset import ForceSpeedDatasetNetx as ForceSpeedDataset


def test_with_netx(net_path, inp, label, sample_length=1000):
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

    return decision, label


def main():
    dataset_path="/media/farscope2/T7 Shield/Neuromorphic Data/George/speed_depth_preproc_downsampled/"

    # Load testing set
    testing_set = ForceSpeedDataset(
        dataset_path,
        train=False,
    )

    start_time = time.time()

    for idx in range(len(testing_set)):
        input, label, filename = testing_set.__getitem__(idx)

        netx_pred, netx_real = test_with_netx(
            net_path="/home/farscope2/Documents/PhD/lava_loihi/networks/best_arm_network/network.net",
            # dataset_path=dataset_path,
            inp=input,
            label=label,
            sample_length=1000
        )

        torch_pred, torch_real = test_with_torch(
            net_path="/home/farscope2/Documents/PhD/lava_loihi/networks/best_arm_network/network.pt",
            inp=input,
            label=label,
            sample_length=1000,
        )


if __name__ == "__main__":
    main()
