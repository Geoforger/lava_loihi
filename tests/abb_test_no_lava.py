import numpy as np
import time
import logging

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import ABBController

from dv_processing import noise
from datetime import timedelta
from lava.lib.dl import netx
from lava.proc import embedded_io as eio
import sys
sys.path.append("..")
from lava_loihi.components.inivation import InivationCamera as Camera
from lava_loihi.components.CustomInivationEncoder import (
    CustomInivationEncoder as CamEncoder,
)
from lava_loihi.components.threshold_pooling import ThresholdPooling as Pooling
from lava_loihi.components.decisions import DecisionMaker
from lava_loihi.components.DecisionVisualiser import DecisionVisualiser as DecisionVis
from lava_loihi.components.DecisionVisualiser import VisualiserWindow
from lava.magma.core.run_conditions import RunContinuous, RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.proc.io.sink import RingBuffer
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
import tkinter as tk
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.runtime.runtime import Runtime


def make_robot():
    return AsyncRobot(SyncRobot(ABBController(ip="192.168.125.1")))

def test_sample(
        target_sample,
        net_path,
        lookup_path,
        tap_move,
        obj_poses,
        home_pose,
        work_frame,
        robot_tcp,
        linear_speed,
        angular_speed,
    ):
    # Setup all loihi parameters
    print("Setting up Loihi...")
    cam_shape = (240, 180)
    filter = noise.BackgroundActivityNoiseFilter(
        cam_shape, backgroundActivityDuration=timedelta(milliseconds=10)
    )
    camera = Camera(noise_filter=filter, flatten=False, crop_params=[20, 2, 50, 30])

    # Initialise network
    net = netx.hdf5.Network(
        net_config=net_path,
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
    in_adapter = eio.spike.PyToNxAdapter(shape=net.inp.shape)
    out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape)
    decision = DecisionMaker(in_shape=out_adapter.out.shape, offset=10, threshold=0.15)

    root = tk.Tk()
    window = VisualiserWindow(root)
    decision_vis = DecisionVis(net_out_shape=net.out.shape, window=window, frequency=10)
    # sink = RingBuffer(shape=camera.s_out.shape, buffer=6000)

    # Connect all components
    camera.s_out.connect(pooling.a_in)
    pooling.s_out.connect(cam_encoder.a_in)
    cam_encoder.s_out.connect(in_adapter.inp)
    in_adapter.out.connect(net.inp)
    net.out.connect(out_adapter.inp)

    # camera.s_out.connect(sink.a_in)

    out_adapter.out.connect(decision.a_in)
    # Connect decision maker ports
    # decision.s_out.connect(sink.a_in)
    decision.s_out.connect(decision_vis.a_in)
    decision_vis.acc_in.connect_var(decision.accumulator)
    decision_vis.conf_in.connect_var(decision.confidence)

    # Set sim parameters
    run_cfg = Loihi2HwCfg(select_tag="fixed_pt")
    run_condition = RunContinuous()

    net._log_config.level = logging.INFO
    net._log_config.level_console = logging.INFO

    compiler = Compiler()
    executable = compiler.compile(camera, run_cfg=run_cfg)
    mp = ActorType.MultiProcessing
    runtime = Runtime(exe=executable,
                    message_infrastructure_type=mp)
    runtime.initialize()
    # runtime.start(run_condition=run_condition)
    # runtime.pause()
    print("Loihi runtime intialised...")

    # Start arm movements
    with make_robot() as robot:
        robot.tcp = robot_tcp
        robot.linear_speed = 50
        robot.angular_speed = 50
        print("Robot info: {}".format(robot.info))
        print("Initial pose in work frame: {}".format(robot.pose))

        print("Moving to origin of work frame ...")
        robot.coord_frame = work_frame
        robot.move_linear((0, 0, 0, 0, 0, 0))
        print("Robot at work frame")

        print("Moving to pose...")
        robot.move_linear(obj_poses[0][0]) # TODO: Make this dynamic
        print("Arm at pose")

        # Tap down
        robot.linear_speed = linear_speed
        robot.angular_speed = angular_speed
        print("Initiating tap...")
        robot.move_linear(tap_move[0])
        print("Tap intiated")

        # Start loihi sim
        # NOTE: Is this blocking?
        runtime.start(run_condition=run_condition)
        # net.run(condition=run_condition, run_cfg=run_cfg)
        time.sleep(3)

        # Slide (blocking)
        print("Sliding...")
        robot.async_move_linear(tap_move[1])
        robot.async_result()

        # Stop sim here
        runtime.pause()
        print("Finished slide")
        # net.stop()
        print("Finished network...")

        # TODO: Get accumulator data

        # Move up
        print("Moving up...")
        robot.move_linear(tap_move[2])
        print("Moved up")

        print("Moving to work frame ...")
        robot.coord_frame = work_frame
        robot.linear_speed = 50
        robot.angular_speed = 50
        robot.move_linear(home_pose)

        # TODO: Implement control scheme
        # linear_speed = change_speed(lookup_path, accumulator, np.arange(5,65,5))


# def change_speed(lookup_path, accumulators, speeds):
#     # Check if decision made

#     if:
#         lookup_table = np.load(lookup_path)
#         # Change decision of not
#         speed_idx = np.argmax(lookup_table[idx_1, idx_2, :])

#         # If new speed required
#         return speeds[speed_idx]
#     else:
#         return False


def main():
    target_label = "Mesh"

    textures = {
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
    net_path = "/home/farscope2/Documents/PhD/lava_loihi/networks/trained_davis_net.net"
    lookup_path="/home/farscope2/Documents/PhD/lava_loihi/data/dataset_analysis/tex_speed_similarity_data.npy"

    robot_tcp=[0, 0, 101.5, 0, 0, 0]
    base_frame=[0, 0, 0, 0, 0, 0]
    home_pose=[400, 0, 240, 180, 0, 180]
    work_frame=[465, -200, 30, 180, 0, 180]
    contact_z = work_frame[2] - textures[target_label]
    obj_poses = ([[0, 0, contact_z, 0, 0, 0], [102, 0, 0.3, 0, 0, 0]],)
    tap_move = [
        [0, 0, contact_z + 1.7, 0, 0, 0],
        [0, 50, contact_z + 1.7, 0, 0, 0],
        [0, 50, 0, 0, 0, 0],
    ]

    # TODO: Test
    test_sample(
        target_sample=target_label,
        net_path=net_path,
        lookup_path=lookup_path,
        tap_move=tap_move,
        obj_poses=obj_poses,
        home_pose=home_pose,
        work_frame=work_frame,
        robot_tcp=robot_tcp,
        linear_speed=10,
        angular_speed=10,
    )

if __name__ == "__main__":
    main()
