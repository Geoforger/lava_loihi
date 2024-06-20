import numpy as np
from lava.lib.dl import netx
from lava.proc.io.source import RingBuffer as InBuffer
from lava.proc.io.sink import RingBuffer as OutBuffer
from lava.proc import embedded_io as eio
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
import logging
CompilerOptions.verbose = True


def test(sparse, buffer_input=None):
    ##############################
    # Rerun with sparse flag True
    ##############################
    run_steps = 1000
    run_cfg = Loihi2HwCfg()

    # Initialise a network
    net = netx.hdf5.Network(
        net_config="/home/farscope2/Documents/PhD/lava_loihi/networks/initial_network_150.net",
        sparse_fc_layer=sparse,
    )
    print(net)

    # Other components
    in_adapter = eio.spike.PyToNxAdapter(shape=net.inp.shape)
    out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape)

    if buffer_input is None:
        buffer_shape = net.inp.shape + (run_steps,)
        # buffer_input = np.ones(buffer_shape)
        buffer_input = np.random.randint(0, 2, size=buffer_shape)
        print(buffer_input)

    source = InBuffer(data=buffer_input)
    sink = OutBuffer(shape=out_adapter.out.shape, buffer=run_steps)

    source.s_out.connect(in_adapter.inp)
    in_adapter.out.connect(net.inp)
    net.out.connect(out_adapter.inp)
    out_adapter.out.connect(sink.a_in)

    net._log_config.level = logging.INFO
    net._log_config.level_console = logging.INFO

    print("Running Network...")
    net.run(condition=RunSteps(num_steps=run_steps), run_cfg=run_cfg)
    data = sink.data.get()
    print("Finished Running")
    net.stop()

    return data, buffer_input


def main():
    # Run test with sparse flag off
    print("Testing dense network...")
    not_sparse_data, buffer_input = test(sparse=False)

    # Rerun with sparse flag on
    print("Testing sparse network...")
    sparse_data, _ = test(sparse=True, buffer_input=buffer_input)

    # Compare output arrays
    print("Dense Data")
    print(not_sparse_data)
    print(not_sparse_data.shape)
    print(np.unique(not_sparse_data))

    print("Sparse data")
    print(sparse_data)
    print(sparse_data.shape)
    print(np.unique(sparse_data))

    both_same = (not_sparse_data == sparse_data).all()
    print(f"Are both the same?: {both_same}")


if __name__ == "__main__":
    main()
