import numpy as np
import pandas as pd
import glob
from scipy.stats import entropy
from simulator_dataloader import SimulatorDataset
from lava.lib.dl import netx
import lava.lib.dl.slayer as slayer
from lava.proc import embedded_io as eio
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.proc.io.source import RingBuffer as InBuffer
from lava.proc.io.sink import RingBuffer as OutBuffer
# from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
# CompilerOptions.verbose = True
import sys
sys.path.append("..")
from utils.utils import nums_from_string

class ControlSimulator():
    def __init__(
        self,
        loihi=False,
        sim_label=0,
        timeout=5,
        lookup_path="",
        network_path="",
        dataset_path="",
        output_path="",
        speeds=[15, 25, 35, 45, 55],
        sample_length = 1000,
    ) -> None:

        # Init hyperparams
        self.loihi = loihi
        self.sim_label = sim_label
        self.timeout = timeout
        self.network_path = network_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.speeds = speeds
        self.sample_length = sample_length
        self.dataset = SimulatorDataset(
            dataset_path, label=sim_label, sampling_time=1
        )

        self.speed, self.lookup_table = self.find_starting_speed(lookup_path)

        # Create output dataframe for logging
        self.data = pd.DataFrame(
            columns=[
                "Filename",
                "Arm Speed",
                "Target Label",
                "Decision",
                "Confidence",
                "Entropy",
                "Total Spikes",
                "Attempt",
            ]
        )

        # Find the number of files in the output dir
        self.test_num = len(
            glob.glob(f"{self.output_path}/{self.sim_label}-iteration-*.csv")
        )

        # Init values used in sim
        self.sim_output = None
        self.accumulator = None
        self.filename = None
        self.current_attempt = 1

    def find_starting_speed(self, lookup_path) -> tuple:
        """
        Method to find starting speed that gives the highest average distance in lookup table
        """
        lookup_table = np.load(lookup_path)
        mean_dist = np.mean(lookup_table, axis=(0, 1))
        speed_idx = np.argmax(mean_dist)

        return self.speeds[speed_idx], lookup_table

    def state_change(self) -> None:
        """
        Method to use the lookup table to find the new exploratory speed for the robot
        """
        arg_sort = np.argsort(self.accumulator)
        highest = arg_sort[-1]
        second = arg_sort[-2]

        # # Check not out of bounds
        # if highest > lookup_table.shape[0]:
        #     if second != 9:
        #         highest = second + 1
        #     elif second != 0:
        #         highest = second - 1

        # if second > lookup_table.shape[0]:
        #     if highest != 9:
        #         second = highest + 1
        #     elif highest != 0:
        #         second = highest - 1

        speed_idx = np.argmax(self.lookup_table[highest, second, :])
        self.speed = self.speeds[speed_idx]

    def test_with_netx(self) -> None:
        """
        Method to run a random file through the simulation
        """
        net = netx.hdf5.Network(net_config=self.network_path)
        run_condition = RunSteps(num_steps=self.sample_length)

        inp, self.filename = self.dataset.__getitem__(speed=self.speed)
        input_data = inp.squeeze()

        input_buffer = InBuffer(data=input_data.numpy())
        output_buffer = OutBuffer(shape=net.out.shape, buffer=self.sample_length)

        if not self.loihi:
            input_buffer.s_out.connect(net.inp)
            net.out.connect(output_buffer.a_in)
            run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
        else:
            in_adapter = eio.spike.PyToNxAdapter(shape=net.inp.shape)
            out_adapter = eio.spike.NxToPyAdapter(shape=net.out.shape)
            input_buffer.s_out.connect(in_adapter.inp)
            in_adapter.out.connect(net.inp)
            net.out.connect(out_adapter.inp)
            out_adapter.out.connect(output_buffer.a_in)
            run_cfg = Loihi2HwCfg(select_tag="fixed_pt")

        #################
        # Run Simulation
        #################
        net.run(condition=run_condition, run_cfg=run_cfg)
        out = output_buffer.data.get()
        net.stop()

        # Maintain sim values for use in logger
        self.sim_output = out
        self.accumulator = np.sum(out, axis=1)

    def log_sim(self, attempt) -> None:
        """
        Method to take the output from the simulation and log it into a csv
        """
        # Calculate metrics per ts from sim output
        total_spikes = np.sum(self.sim_output, axis=0)
        decision = np.argmax(self.sim_output, axis=0)
        highest_spikes = np.max(self.sim_output, axis=0)
        confidence = highest_spikes / total_spikes
        entropy = self.__calculate_entropy(highest_spikes)

        # Append test values to dataframe
        inp = {
            "Filename": np.repeat(self.filename, self.sample_length),
            "Arm Speed": np.repeat(self.speed, self.sample_length),
            "Target Label": np.repeat(self.sim_labelm, self.sample_length),
            "Decision": decision,
            "Confidence": confidence,
            "Entropy": entropy,
            "Total Spikes": total_spikes,
            "Attempt": np.repeat(attempt, self.sample_length),
        }

        self.data = pd.concat(
            [self.data, pd.DataFrame(inp)], ignore_index=True
        )

        # Save dataframe to csv at end of testing
        if self.current_attempt == self.timeout:
            self.data.to_csv(
                f"{self.output_path}/{self.sim_label}-iteration-{self.test_num}.csv"
            )
        else:
            self.current_attempt += 1

    def __calculate_entropy(self) -> float:
        """
        Private method to calculate entropy at each time step
        """
        entropy_values = np.zeros(self.sample_length, dtype=float)

        # Cumulative sum of spikes over time for each neuron
        cumulative_spikes = np.cumsum(self.sim_output, axis=1)

        # Cumulative total spikes across all neurons at each time step
        total_spikes = cumulative_spikes.sum(axis=0)

        # Small constant to avoid division by zero
        epsilon = 1e-12

        for ts in range(self.sample_length):
            # Spike rates up to current time step for each neuron
            spike_rates = cumulative_spikes[:, ts]

            # Normalize to get probabilities
            total = total_spikes[ts] + epsilon  # Avoid division by zero
            probabilities = spike_rates / total

            # Compute entropy
            entropy_ts = entropy(probabilities, base=2)
            entropy_values[ts] = entropy_ts

        return entropy_values

    def run_simulation(self):
        for attempt in range(self.timeout):
            self.test_with_netx()
            self.log_sim(attempt)
            self.state_change()

def main():
    #################
    # Simulator hyperparams
    #################
    sim = ControlSimulator(
        loihi = False,
        sim_label = 0,
        timeout = 5,
        lookup_path = "",
        network_path = "",
        dataset_path = "",
        output_path = "",
    )

    #################
    # Simulation Loop
    #################
    sim.run_simulation()

if __name__ == "__main__":
    main()
