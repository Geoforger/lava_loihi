import numpy as np

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class ThresholdPooling(AbstractProcess):
    def __init__(
        self,
        in_shape: tuple,
        kernel: tuple,
        stride: int,
        thresold: int,
    ) -> None:

        self.in_shape = in_shape
        self.kernel = kernel
        self.stride = stride
        self.threshold = thresold

        self.pooling_indices = self.__create_pooling_indices()
        self.num_windows = len(self.pooling_indices)
        self.num_x = (self.in_shape[1] - self.kernel[1]) // self.stride[1] + 1

        self.out_shape = self.__calculate_output_shape()

        self.a_in = InPort(shape=self.in_shape)
        self.s_out = OutPort(shape=self.out_shape)

        super().__init__(
            in_shape=self.in_shape,
            out_shape=self.out_shape,
            kernel=self.kernel,
            stride=self.stride,
            threshold=self.threshold,
            pooling_indices=self.pooling_indices,
            num_windows=self.num_windows,
            num_x=self.num_x
        )

    def __create_pooling_indices(self):
        cols, rows = self.in_shape
        k_cols, k_rows = self.kernel
        s_cols, s_rows = self.stride

        # Calculate the number of windows in each dimension
        num_windows_x = (cols - k_cols) // s_cols + 1
        num_windows_y = (rows - k_rows) // s_rows + 1

        # List of top-left indices of each pooling window
        window_indices = []
        for y in range(num_windows_y):
            for x in range(num_windows_x):
                window_indices.append((x * s_cols, y * s_rows))

        return window_indices

    def __calculate_output_shape(self):
        x_dim = int(((self.in_shape[0] - self.kernel[0]) / self.stride[0]) + 1)
        y_dim = int(((self.in_shape[1] - self.kernel[1]) / self.stride[1]) + 1)

        return (x_dim, y_dim)


@implements(proc=ThresholdPooling, protocol=LoihiProtocol)
@requires(CPU)
class PyThresholdPoolingModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_SPARSE, int)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)

        self.in_shape = proc_params["in_shape"]
        self.out_shape = proc_params["out_shape"]
        self.kernel = proc_params["kernel"]
        self.stride = proc_params["stride"]
        self.threshold = proc_params["threshold"]
        self.pooling_indices = proc_params["pooling_indices"]
        self.num_windows = proc_params["num_windows"]
        self.num_x = proc_params["num_x"]

    def run_spk(self):
        # Get spike information
        data, indices = self.a_in.recv()

        if len(data) > 0:
            # Unravel into 2D indices
            spike_ys, spike_xs = np.unravel_index(indices, self.in_shape)

            # Compute window indices in a vectorized manner
            window_idx = self.__find_window_idx_vectorized(spike_xs, spike_ys)

            # Initialize counters for spike types
            count_ones = np.zeros(len(self.pooling_indices), dtype=int)
            count_zeros = np.zeros(len(self.pooling_indices), dtype=int)

            # Update counts based on spike data
            count_ones += np.bincount(
                window_idx[data == 1], minlength=len(self.pooling_indices)
            )
            count_zeros += np.bincount(
                window_idx[data == 0], minlength=len(self.pooling_indices)
            )

            # Initialize output mapping with 2s (default state)
            output_mapping = np.full(len(self.pooling_indices), 2, dtype=int)

            # Update output mapping where thresholds are surpassed
            output_mapping[count_ones >= self.threshold] = 1
            output_mapping[count_zeros >= self.threshold] = 0

            # Find valid indices from output mapping
            valid_indices = np.where(
                (output_mapping == 1) | (output_mapping == 0))[0]

            # Send out the data and indices
            self.s_out.send(output_mapping[valid_indices], valid_indices)
        else:
            self.s_out.send(data, indices)

    def __find_window_idx_vectorized(self, spike_xs, spike_ys):
        window_xs = spike_xs // self.stride[1]
        window_ys = spike_ys // self.stride[0]
        window_idx = window_ys * self.num_x + window_xs
        # Ensure indices are within the valid range
        window_idx = np.clip(window_idx, 0, len(self.pooling_indices) - 1)
        return window_idx
