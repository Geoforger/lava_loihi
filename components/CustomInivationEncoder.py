import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.variable import Var


class CustomInivationEncoder(AbstractProcess):
    def __init__(self, in_shape, flatten=True, off_events=False) -> None:

        self.off_events = off_events
        self.flatten = flatten
        self.in_shape = in_shape

        if flatten:
            cam_dims = (np.prod(in_shape),)
        else:
            cam_dims = in_shape

        if not self.off_events:
            self.out_shape = (1,) + cam_dims
        else:
            self.out_shape = (2,) + cam_dims

        self.a_in = InPort(shape=self.in_shape)
        self.s_out = OutPort(shape=self.out_shape)

        super().__init__(
            in_shape = self.in_shape,
            out_shape = self.out_shape,
            flatten = self.flatten,
            off_events = self.off_events
        )


@tag("fixed_pt")
@implements(proc=CustomInivationEncoder, protocol=LoihiProtocol)
@requires(CPU)
class PyCustomInivationEncoderModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.in_shape = proc_params["in_shape"]
        self.out_shape = proc_params["out_shape"]
        self.flatten = proc_params["flatten"]
        self.off_events = proc_params["off_events"]

    def run_spk(self) -> None:
        polarities, indices = self.a_in.recv()

        on_events = indices[np.where(polarities == 1)]

        if not self.off_events:
            output_arr = np.zeros(self.out_shape)
            output_arr[0, on_events] = 1

        # TODO: Implement option for off events
        # NOTE: This is currently untested and may not be correct implementation
        else:
            off_events = indices[np.where(polarities == 0)]
            output_arr = np.zeros(self.out_shape)
            output_arr[0, off_events] = 1
            output_arr[1, on_events] = 1

        self.s_out.send(output_arr)
