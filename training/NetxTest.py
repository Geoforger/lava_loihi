import numpy as np
import logging
from lava.lib.dl import netx
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io.source import RingBuffer as InBuffer
from lava.proc.io.sink import RingBuffer as OutBuffer
from torch.utils.data import DataLoader

from force_speed_dataset import ForceSpeedDataset
from sklearn.metrics import accuracy_score

def test_with_netx(net_path, dataset_path, sample_length=1000):
    # Import net
    net = netx.hdf5.Network(
        net_config=net_path,
        sparse_fc_layer=False
    )
    
    # Load testing set
    testing_set = ForceSpeedDataset(
        dataset_path,
        train=False,
    )
    
    test_loader  = DataLoader(dataset=testing_set, batch_size=1, shuffle=True)
    
    # Sim settings
    run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
    run_condition = RunSteps(num_steps=sample_length)
    net._log_config.level = logging.INFO
    net._log_config.level_console = logging.INFO

    pred_labels = []
    real_labels = []
    
    for _, (input, label, speed, force) in enumerate(test_loader):
        print(input)
        print(input.shape)
        
        # Ring buffer with data
        input_buffer = InBuffer(data=input)
        output_buffer = OutBuffer(shape=net.out.shape, buffer=sample_length)
        
        # Connect to network
        input_buffer.s_out.connect(net.inp)
        net.out.connect(output_buffer.a_in)
        
        # Run sim
        print("Running Network...")
        net.run(condition=run_condition, run_cfg=run_cfg)
        out = output_buffer.data.get()
        
        print(out)
        print(out.shape)
        
        # Classify based on max spikes
        output_totals = np.sum(out)
        pred_labels.append(np.argmax(output_totals))
        real_labels.append(label)
        
    print(f"Accracy across test set: {accuracy_score(pred_labels, real_labels)}")
        