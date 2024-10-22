import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import glob
from scipy.stats import entropy
from simulator_dataloader import SimulatorDataset
from lava.lib.dl import netx
import lava.lib.dl.slayer as slayer
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.proc.io.source import RingBuffer as InBuffer
from lava.proc.io.sink import RingBuffer as OutBuffer

import sys
sys.path.append("..")
from utils.utils import nums_from_string


class ControlSimulator():
    def __init__(
        self,
        mode,
        lookup_path,
        network_path,
        dataset_path,
        output_path,
        sim_label,        
        loihi=False,
        speeds=[15, 25, 35, 45, 55],
        sample_length = 1000,
    ) -> None:

        # Init hyperparams
        self.mode = mode
        self.loihi = loihi
        self.sim_label = sim_label
        self.network_path = network_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.speeds = speeds
        self.sample_length = sample_length
        self.dataset = SimulatorDataset(
            dataset_path, label=sim_label, sampling_time=1
        )
        
        if self.loihi:
            from lava.proc import embedded_io as eio
            from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
            CompilerOptions.verbose = True

        # start_speed, self.lookup_table = self.find_starting_speed(lookup_path)
        start_speed, self.lookup_table = self.random_start_speed(lookup_path)
        self.attempt_speeds = [start_speed]

        # Find the number of files in the output dir
        self.test_num = len(
            glob.glob(f"{self.output_path}/{self.mode}-{self.sim_label}-iteration-*.csv")
        )

        # Init values used in sim
        self.data = None
        self.sim_output = None
        self.filenames = []
        self.current_attempt = 1
        self.pred = None

    def find_starting_speed(self, lookup_path) -> tuple:
        """
        Method to find starting speed that gives the highest average distance in lookup table
        """
        lookup_table = np.load(lookup_path)
        mean_dist = np.mean(lookup_table, axis=(0, 1))
        speed_idx = np.argmax(mean_dist)

        return self.speeds[speed_idx], lookup_table

    def random_start_speed(self, lookup_path) -> tuple:
        """
        Method to return lookup table and a random starting speed
        """
        lookup_table = np.load(lookup_path)
        return np.random.choice(self.speeds), lookup_table

    def state_change(self) -> None:
        """
        Method to use the lookup table to find the new exploratory speed for the robot
        based on the currently predicted texture
        """
        speed_idx = np.argmin(self.lookup_table[self.pred])
        try:
            self.attempt_speeds.append(self.speeds[speed_idx])
        except Exception as e:
            print(e)
            print(speed_idx)
            print(self.pred)
            print(self.lookup_table[self.pred])
            
    def von_rossum_change(self) -> None:
        """
        Method to use the lookup table to find the new exploratory speed for the robot
        """
        arg_sort = np.argsort(self.acc[:, -1])
        highest = arg_sort[-1]
        second = arg_sort[-2]

        speed_idx = np.argmax(self.lookup_table[highest, second, :])
        self.attempt_speeds.append(self.speeds[speed_idx])

    def rand_change(self) -> None:
        speed = np.random.choice(self.speeds)
        self.attempt_speeds.append(speed)

    def test_with_netx(self) -> None:
        """
        Method to run a random file through the simulation
        """
        net = netx.hdf5.Network(net_config=self.network_path)
        run_condition = RunSteps(num_steps=self.sample_length)

        inp, f = self.dataset.__getitem__(speed=self.attempt_speeds[-1])
        self.filenames.append(f)
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

    def log_sim(self, save=False) -> None:
        """
        Method to take the output from the simulation and log it into a csv
        """
        # Analyse this sample
        test_data = self.sim_output
        
        self.acc = np.cumsum(test_data, axis=1)
        total_spikes = np.sum(self.acc, axis=0)
        decision = np.argmax(self.acc, axis=0)
        self.pred = decision[-1]
        highest_spikes = np.max(self.acc, axis=0)
        confidence = highest_spikes / total_spikes
        entropy = self.__calculate_entropy(total_spikes)
        # entropy = self.__calculate_moving_window_entropy(acc)
        
        # Meta params for each attempt
        # NOTE: These need reiterating at each timestep in output data
        f = np.repeat(self.filenames[-1], self.sample_length)
        s = np.repeat(self.attempt_speeds[-1], self.sample_length)
        a = np.repeat(self.current_attempt, self.sample_length)
        
        # Create output dataframe
        inp = {
            "Mode": self.mode,
            "Filename": f,
            "Time Step": np.arange(self.sample_length),
            "Arm Speed": s,
            "Target Label": np.repeat(self.sim_label, self.sample_length),
            "Decision": decision,
            "Confidence": confidence,
            "Entropy": entropy,
            "Total Spikes": total_spikes,
            "Max Spikes": highest_spikes,
            "Attempt": a,
        }
        save_data = pd.DataFrame(data=inp)
        if self.current_attempt == 1:
            self.data = save_data
        else:
            self.data = pd.concat([self.data, save_data], ignore_index=True)

        # Save dataframe to csv at end of testing
        if save:
            print("Saving data...")
            self.data.to_csv(
                f"{self.output_path}/{self.mode}-{self.sim_label}-iteration-{self.test_num}.csv"
            )

    def __calculate_entropy(self, total_spikes) -> float:
        """
        Private method to calculate entropy at each time step
        """
        entropy_values = np.zeros(self.sample_length, dtype=float)

        # Small constant to avoid division by zero
        epsilon = 1e-12

        for ts in range(self.sample_length):
            # Spike rates up to current time step for each neuron
            spike_rates = self.acc[:, ts]

            # Normalize to get probabilities
            total = total_spikes[ts] + epsilon  # Avoid division by zero
            probabilities = spike_rates / total

            # Compute entropy
            entropy_ts = entropy(probabilities, base=2)
            entropy_values[ts] = entropy_ts

        return entropy_values

    def __calculate_moving_window_entropy(
        self, cumulative_data, window_size=10, pseudocount=1e-6, spike_threshold=10

    ):
        """
        Calculate entropy at each time step using cumulative spike counts.

        Parameters:
        - cumulative_data: 2D numpy array of shape (n_neurons, n_time_steps)
                        Cumulative sum of spikes for each neuron up to each time step.
        - pseudocount: Small float added to spike counts to prevent zero probabilities.
        - spike_threshold: Minimum total cumulative spike count to compute entropy.
                        If the total spikes are below this threshold, entropy is set to NaN.

        Returns:
        - entropy_over_time: 1D numpy array of entropy values over time.
                            Length is n_time_steps.
        """
        n_neurons, n_time_steps = cumulative_data.shape

        # Initialize array to hold entropy values
        entropy_over_time = np.full(n_time_steps, np.nan)

        for ts in range(n_time_steps):
            # Determine the start index of the window
            start_idx = max(0, ts - window_size + 1)
            end_idx = ts

            # Calculate spikes within the window for each neuron
            if start_idx == 0:
                window_spike_counts = cumulative_data[:, end_idx]
            else:
                window_spike_counts = cumulative_data[:, end_idx] - cumulative_data[:, start_idx - 1]

            # Add pseudocount to avoid zero probabilities
            spike_counts = window_spike_counts + pseudocount

            # Total spikes in the window
            total_spikes = spike_counts.sum()

            if total_spikes - pseudocount * n_neurons < spike_threshold:
                # Not enough spikes to compute entropy reliably
                entropy_over_time[ts] = np.nan
            else:
                # Normalize to get probabilities
                probabilities = spike_counts / total_spikes

                # Compute entropy using scipy's entropy function
                entropy_value = entropy(probabilities, base=2)
                entropy_over_time[ts] = entropy_value

        return entropy_over_time

    def run_simulation(self):
        self.test_with_netx()
        self.log_sim(save=False)
        
        # If random mode select a random speed to switch to
        if self.mode == "r":
            self.rand_change()
        # If test mode implement the correct switch
        elif self.mode == "t":
            self.state_change()
        # Else keep using the starting speed and append to list again
        elif self.mode == "b":
            self.attempt_speeds.append(self.attempt_speeds[-1])
        elif self.mode == "v":
            self.von_rossum_change()

        self.current_attempt += 1
        self.test_with_netx()
        self.log_sim(save=True)

def main():
    #################
    # Simulator hyperparams
    #################
    mode = sys.stdin.readline().strip()
    target_label = int(sys.stdin.readline().strip())
    
    if mode != "v":
        lookup_path =  "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/labels_speeds_entropy.npy"
    else:
        lookup_path =  "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/tex_tex_speed_similarity_data.npy"
    
    sim = ControlSimulator(
        mode = mode,
        loihi = False,
        sim_label = target_label,
        lookup_path=lookup_path,
        network_path = "/media/george/T7 Shield/Neuromorphic Data/George/arm_networks/spike_max_arm_test_1729473122/network.net",
        dataset_path = "/media/george/T7 Shield/Neuromorphic Data/George/simulator_testing/",
        output_path = "/media/george/T7 Shield/Neuromorphic Data/George/tests/simulator_tests/",
    )

    #################
    # Simulation Loop
    #################
    sim.run_simulation()

if __name__ == "__main__":
    main()
