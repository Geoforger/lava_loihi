import numpy as np
from dv_processing import noise
from datetime import timedelta
import logging
import time
import tkinter as tk
from lava.lib.dl import netx
from lava.proc import embedded_io as eio

import sys

sys.path.append("..")
from lava_loihi.components.inivation import InivationCamera as Camera
from lava_loihi.components.CustomInivationEncoder import (
    CustomInivationEncoder as CamEncoder,
)
from lava_loihi.components.iniviation_visualiser import InivationVisualiser as Vis
from lava_loihi.components.threshold_pooling import ThresholdPooling as Pooling
from lava_loihi.components.decisions import DecisionMaker
from lava_loihi.components.DecisionVisualiser import DecisionVisualiser as DecisionVis
from lava_loihi.components.DecisionVisualiser import VisualiserWindow
from lava_loihi.components.ABBController import ABBController
from lava_loihi.components.Datalogger import Datalogger
from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
# from lava.proc.io.sink import RingBuffer
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions

CompilerOptions.verbose = True

def main():
    run_steps = 100000
    sim_time = 120
    loihi = False
    target_label = 0
    output_path = "/home/farscope2/Documents/PhD/lava_loihi/data/arm_tests/"

    # Camera init
    cam_shape = (240, 180)
    filter = noise.BackgroundActivityNoiseFilter(
        cam_shape, backgroundActivityDuration=timedelta(milliseconds=10)
    )
    camera = Camera(noise_filter=filter, flatten=False, crop_params=[20, 2, 50, 30])

    # Initialise network
    net = netx.hdf5.Network(
        net_config="/home/farscope2/Documents/PhD/lava_loihi/networks/trained_davis_net.net",
        sparse_fc_layer=True,
    )
    print(net)

    # Init other components
    pooling = Pooling(
        in_shape=camera.out_shape, kernel=(4, 4), stride=(4, 4), threshold=1
    )
    # NOTE: I have an encoder that takes camera -> dense -> nx encoder
    #       This could be skipped if I could be bothered to code in C
    cam_encoder = CamEncoder(pooling.s_out.shape)
    input_vis = Vis(in_shape=pooling.out_shape, flattened_input=False)

    texture_depths = {
        "Mesh": 20.7,
        "Felt": 22.2,
        "Cotton": 20.4,
        "Nylon": 20.4,
        "Fur": 22.5,
        "Wood": 20.5,
        "Acrylic": 19.8,
        "FashionFabric": 19.8,
        "Wool": 22.2,
        "Canvas": 20.0,
    }

    abb_params = {
        "robot_tcp": [0, 0, 101.5, 0, 0, 0],
        "base_frame": [0, 0, 0, 0, 0, 0],
        "home_pose": [400, 0, 240, 180, 0, 180],
        "work_frame": [465, -200, 30, 180, 0, 180],
        "tap_depth": 1.5,
        "tap_length": 50,
    }

    root = tk.Tk()
    window = VisualiserWindow(root)
    decision_vis = DecisionVis(net_out_shape=net.out.shape, window=window, frequency=10)

    # Sim vs Loihi setup
    if loihi is True:
        in_adapter = eio.spike.PyToNxAdapter(shape=net.inp.shape)
        out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape)
        decision = DecisionMaker(
            in_shape=out_adapter.out.shape, offset=10, threshold=0.15
        )

        run_cfg = Loihi2HwCfg(select_tag="fixed_pt")
        # Connect all components
        camera.s_out.connect(pooling.a_in)
        pooling.s_out.connect(input_vis.a_in)
        pooling.s_out.connect(cam_encoder.a_in)
        cam_encoder.s_out.connect(in_adapter.inp)
        in_adapter.out.connect(net.inp)
        net.out.connect(out_adapter.inp)
        out_adapter.out.connect(decision.a_in)
    else:
        decision = DecisionMaker(in_shape=net.out.shape, offset=10, threshold=0.15)

        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        # Connect all components
        camera.s_out.connect(pooling.a_in)
        pooling.s_out.connect(input_vis.a_in)
        pooling.s_out.connect(cam_encoder.a_in)
        cam_encoder.s_out.connect(net.inp)
        net.out.connect(decision.a_in)

    # Connect decision maker ports
    decision.s_out.connect(decision_vis.a_in)
    decision_vis.acc_in.connect_var(decision.accumulator)
    decision_vis.conf_in.connect_var(decision.confidence)

    # Arm controller and connections
    abb_controller = ABBController(
        net_out_shape=net.out.shape,
        lookup_path="/home/farscope2/Documents/PhD/lava_loihi/data/dataset_analysis/tex_speed_similarity_data.npy",
        textures=texture_depths,
        speeds=np.arange(5, 65, 5),
        abb_params=abb_params,
        target_texture="Mesh",
        timeout=5
    )
    abb_controller.acc_in.connect_var(decision.accumulator)

    # Datalogger and connections
    logger = Datalogger(
        out_path=output_path,
        net_out_shape=net.out.shape,
        target_label=target_label,
    )
    logger.conf_in.connect_var(decision.confidence)
    # logger.decision_in.connect_var(decision.decision)
    logger.attempt_in.connect_var(abb_controller.attempt)
    logger.arm_speed_in.connect_var(abb_controller.slide_speed)

    # Set sim parameters
    run_condition = RunContinuous()
    # run_condition = RunSteps(num_steps=run_steps)

    net._log_config.level = logging.INFO
    net._log_config.level_console = logging.INFO

    print("Running Network...")
    net.run(condition=run_condition, run_cfg=run_cfg)
    print("Started sim..")
    time.sleep(sim_time)  # TODO: How to stop sim early
    net.stop()
    print("Finished running")


if __name__ == "__main__":
    main()
