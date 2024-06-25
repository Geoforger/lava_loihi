import numpy as np

from lava.magma.core.decorator import implements, requires
from lava.magma.core.process.variable import Var
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyVarPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, VarPort, RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class DecisionVisualiser(AbstractProcess):
    """
    Process that visualises the output of our decision making algorithm

    Parameters:
    -----------
        window_name (string, optional):
            Name of the CV window. Defaults 'Decision Visualiser'
    Returns:
        None
    """

    def __init__(
        self, 
        net_out_shape,
        window_name="Decision Visualiser"
    ) -> None:

        self.in_shape = (1,)
        self.net_out_shape = net_out_shape
        self.window_name = window_name

        self.a_in = InPort(shape=self.in_shape)
        self.conf_in = RefPort((1,))
        self.acc_in = RefPort(self.net_out_shape)

        # Values to take from Var ports
        self.acc = None
        self.conf = None
        self.decision = None

        super().__init__(
            in_shape=self.in_shape,
            net_out_shape=self.net_out_shape,
            window_name=self.window_name,
            acc = self.acc,
            conf = self.conf,
            decision = self.decision
        )


@implements(proc=DecisionVisualiser, protocol=LoihiProtocol)
@requires(CPU)
class PySparseDecisionVisualiserModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    acc_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int64)
    conf_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)

    # acc: np.ndarray = LavaPyType(np.ndarray, np.int64)
    # conf: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.in_shape = proc_params["in_shape"]
        self.net_out_shape = proc_params["net_out_shape"]
        self.window_name = proc_params["window_name"]
        self.acc = proc_params["acc"]
        self.conf = proc_params["conf"]
        self.decision = proc_params["decision"]

    def update_plot(self):
        pass

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        # Recieve values for accumulator and conf in post mgmt
        self.acc = self.acc_in.read()
        # print(f"Vis Accumulator: {accumulator} : {self.time_step}")
        self.conf = self.conf_in.read()

    def run_spk(self) -> None:
        # NOTE: Due to run_post_mgmt, we are 1ts out of sync with network
        if (self.acc is not None) & (self.conf != None):
            # print(f"Vis Accumulator: {self.acc} : {self.time_step}")
            # print(f"Vis Conf: {self.conf} : {self.time_step}")
            pass

        # Recieve decision for next timestep
        self.decision = self.a_in.recv()
