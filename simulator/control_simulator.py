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
        timeout=5,
        speeds=[15, 25, 35, 45, 55],
        sample_length = 1000,
    ) -> None:

        # Init hyperparams
        self.mode = mode
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
        
        if self.loihi:
            from lava.proc import embedded_io as eio
            from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
            CompilerOptions.verbose = True

        start_speed, self.lookup_table = self.find_starting_speed(lookup_path)
        self.attempt_speeds = [start_speed]

        # Find the number of files in the output dir
        self.test_num = len(
            glob.glob(f"{self.output_path}/{self.sim_label}-iteration-*.csv")
        )

        # Init values used in sim
        self.sim_output = None
        self.all_test_data = np.empty((10, self.timeout*self.sample_length))
        self.acc = None
        self.filenames = []
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

    def log_sim(self, attempt) -> None:
        """
        Method to take the output from the simulation and log it into a csv
        """
        # Add this sample output to the accumulator
        self.all_test_data[:, (self.current_attempt-1)*self.sample_length:self.current_attempt*self.sample_length] = self.sim_output
        self.acc = np.cumsum(self.all_test_data[:, :self.current_attempt*self.sample_length], axis=1)
        
        if self.current_attempt == self.timeout:
            self.acc = np.cumsum(self.all_test_data, axis=1)
            total_spikes = np.sum(self.acc, axis=0)
            decision = np.argmax(self.acc, axis=0)
            highest_spikes = np.max(self.acc, axis=0)
            confidence = highest_spikes / total_spikes
            entropy = self.__calculate_entropy(total_spikes)
            
            # Meta params for each attempt
            # NOTE: These need reiterating at each timestep in output data
            f = [n for name in self.filenames for n in np.repeat(name, self.sample_length)]
            s = [sp for speed in self.attempt_speeds for sp in np.repeat(speed, self.sample_length)]
            a = [at+1 for att in range(self.timeout) for at in np.repeat(att, self.sample_length)]
            
            # Create output dataframe
            inp = {
                "Mode": self.mode,
                "Filename": f,
                "Arm Speed": s,
                "Target Label": np.repeat(self.sim_label, self.sample_length*self.timeout),
                "Decision": decision,
                "Confidence": confidence,
                "Entropy": entropy,
                "Total Spikes": total_spikes,
                "Attempt": a,
            }
            save_data = pd.DataFrame(data=inp)
            
            # Save dataframe to csv at end of testing
            print("Saving data...")
            save_data.to_csv(
                f"{self.output_path}/{self.sim_label}-iteration-{self.test_num}.csv"
            )
        else:
            self.current_attempt += 1

    def __calculate_entropy(self, total_spikes) -> float:
        """
        Private method to calculate entropy at each time step
        """
        entropy_values = np.zeros(self.sample_length * self.timeout, dtype=float)

        # Small constant to avoid division by zero
        epsilon = 1e-12

        for ts in range(self.sample_length * self.timeout):
            # Spike rates up to current time step for each neuron
            spike_rates = self.acc[:, ts]

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
            
            # If random mode select a random speed to switch to
            if self.mode == "r":
                self.rand_change()
            # If test mode implement the correct switch
            elif self.mode == "t":
                self.state_change()
            # Else keep using the starting speed

def main():
    #################
    # Simulator hyperparams
    #################
    mode = sys.stdin.readline().strip()
    target_label = int(sys.stdin.readline().strip())
    
    print(f"Mode: {mode}")
    print(f"Label: {target_label}")

    sim = ControlSimulator(
        mode = mode,
        loihi = False,
        sim_label = target_label,
        timeout = 5,
        lookup_path =  "/media/george/T7 Shield/Neuromorphic Data/George/tests/dataset_analysis/tex_tex_speed_similarity_data.npy",
        network_path = "/media/george/T7 Shield/Neuromorphic Data/George/arm_networks/arm_test_nonorm_1728559746/network.net",
        dataset_path = "/media/george/T7 Shield/Neuromorphic Data/George/speed_depth_preproc_downsampled/",
        output_path = "/media/george/T7 Shield/Neuromorphic Data/George/tests/simulator_tests/",
    )

    #################
    # Simulation Loop
    #################
    sim.run_simulation()

if __name__ == "__main__":
    main()
