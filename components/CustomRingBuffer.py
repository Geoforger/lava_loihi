import numpy as np

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.variable import Var


class CustomRingBuffer(AbstractProcess):
    def __init__(self, shape, buffer) -> None:

        self.shape = shape
        self.buffer = buffer
        self.buffer_shape = (1,) + self.shape + (self.buffer,)

        self.a_in = InPort(shape=self.shape)
        self.data = Var(shape=self.buffer_shape, init=np.zeros(self.buffer_shape))

        super().__init__(
            shape=self.shape,
            buffer=self.buffer,
        )

@implements(proc=CustomRingBuffer, protocol=LoihiProtocol)
@requires(CPU)
class PyCustomRingBufferModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)
    data: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.shape = proc_params["shape"]
        self.buffer = proc_params["buffer"]

    def run_spk(self) -> None:
        polarities, indices = self.a_in.recv()

        on_events = indices[np.where(polarities == 1)]

        on_x_coords, on_y_coords = np.unravel_index(on_events, self.shape)

        self.data[0, on_x_coords, on_y_coords, self.time_step - 1] = 1
