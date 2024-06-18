import numpy as np
from lava.lib.dl import netx
from lava.proc.io.source import RingBuffer
from lava.proc import embedded_io as eio
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
CompilerOptions.verbose = True
import logging


def main():
    run_steps = 1000
    run_cfg = Loihi2HwCfg()

    # Initialise a network
    net = netx.hdf5.Network(
        net_config="/home/farscope2/Documents/PhD/lava_loihi/networks/initial_network_275.net"
    )
    print(net)

    # Other components
    buffer_shape = net.inp.shape + (run_steps,)
    buffer_input = np.ones(buffer_shape)
    source = RingBuffer(data=buffer_input)

    in_adapter = eio.spike.PyToNxAdapter(shape=net.inp.shape)

    source.s_out.connect(in_adapter.inp)
    in_adapter.out.connect(net.inp)

    net._log_config.level = logging.INFO
    net._log_config.level_console = logging.INFO

    print("Running Network...")
    net.run(condition=RunSteps(num_steps=run_steps), run_cfg=run_cfg)
    print("Finished Running")
    net.stop()


if __name__ == "__main__":
    main()
