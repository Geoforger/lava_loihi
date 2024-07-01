import numpy as np
import time
from dv_processing import noise
from datetime import timedelta
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io.sink import RingBuffer
import sys
sys.path.append("..")
from lava_loihi.components.inivation import InivationCamera as Camera


def sim(filter=True, crop_params=None):
    run_steps = 1000

    # Camera init
    cam_shape = (240, 180)
    if filter:
        filter = noise.BackgroundActivityNoiseFilter(
            cam_shape, backgroundActivityDuration=timedelta(milliseconds=10)
        )
    else:
         filter = None
    camera = Camera(noise_filter=filter, flatten=False, crop_params=crop_params)
    sink = RingBuffer(shape=camera.s_out.shape, buffer=run_steps)

    # Setup connections
    camera.s_out.connect(sink.a_in)

    run_cfg = Loihi2SimCfg()
    run_condition = RunSteps(num_steps=run_steps)

    print("Running Network...")
    camera.run(condition=run_condition, run_cfg=run_cfg)
    print("Finished Running")
    camera.stop()

    time.sleep(3)


def main():
    # Crop parameters
    crop_params = [20, 2, 50, 30]

    print("Simulating with no cropping")
    # Sim with filter
    print("Simulating with filter")
    for s in range(1000):
        print(f"Sim: {s}")
        sim()

    # Sim with no filter
    print("Simulating with no filter")
    for s in range(1000):
        print(f"Sim: {s}")
        sim(filter=False)

    print("Simulating with cropping")
    # Sim with filter
    print("Simulating with filter")
    for s in range(1000):
        print(f"Sim: {s}")
        sim(crop_params=crop_params)

    # Sim with no filter
    print("Simulating with no filter")
    for s in range(1000):
        print(f"Sim: {s}")
        sim(filter=False, crop_params=crop_params)

if __name__ == "__main__":
        main()
