import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort, RefPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel


class DecisionMaker(AbstractProcess):
    """Decision making process for real-time classification.

    By accumulating data for each input neuron in the input vector
    this process outputs a classification in real-time.
    The offset and threshold params are used to tune the accuracy
    of the classification. Increases to either values increases accuracy
    at the expense of time.
    > Higher offset values will increase the time required for a
      classification.
    > Threshold values should be tuned based on your offset.

    Parameters
    ----------

    in_shape (int): Shape of input vector.
    offset (OPTIONAL. Default=0) (int):
        Optional param for total information required to output a decision.
    threshold (OPTIONAL. Default=0) (float):
        Optional param for confidence in the gathered information required to
        output a decision.
    prior (OPTIONAL. Default=0) (int):
        Optional param to set the prior of each information accumulator.
        Low values increase sensitivity at the cost of instability.
    """
    def __init__(
        self,
        in_shape: tuple,
        offset: int = 0,
        threshold: float = 0.0,
        prior: int = 0,
    ) -> None:

        self.in_shape = in_shape
        self.offset = offset
        self.threshold = threshold

        # Variables that can be probed must be setup using lava variables
        self.confidence = Var(shape=(1,), init=0.0)
        self.accumulator = Var(
            shape=self.in_shape, init=(np.ones(self.in_shape) * prior)
        )
        self.decision = Var(shape=(1,), init=np.array([-1]))

        self.a_in = InPort(shape=in_shape)
        self.s_out = OutPort(shape=(1,))

        super().__init__(
            in_shape=self.in_shape,
            offset=self.offset,
            threshold=self.threshold,
        )


@implements(proc=DecisionMaker, protocol=LoihiProtocol)
@requires(CPU)
class PyDecisionMaker(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int8)

    accumulator: np.ndarray = LavaPyType(np.ndarray, np.int64)
    confidence: np.ndarray = LavaPyType(np.ndarray, float)
    decision: np.ndarray = LavaPyType(np.ndarray, int)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)

        self.in_shape = proc_params["in_shape"]
        self.offset = proc_params["offset"]
        self.threshold = proc_params["threshold"]

    # Function to run on each timestep
    def run_spk(self):
        # Get input vector
        data_in = self.a_in.recv()
        # Accumulate input data
        self.accumulator = self.accumulator + data_in

        # Perform calculcations
        spikes = np.sum(self.accumulator)
        max_spikes = np.amax(self.accumulator)
        if spikes > 0:
            self.confidence = np.array([max_spikes / spikes])

            if (spikes >= self.offset) & (self.confidence >= self.threshold):
                # NOTE: This returns the first max value in the array
                # if there are multiple instances of the same value present
                self.decision = np.argmax(self.accumulator)
                self.s_out.send(np.array([self.decision]))
                return

        # If not reached an output, send a random guess
        # self.decision = np.random.randint(len(self.accumulator))
        self.decision = -1
        self.decision = np.array([self.decision])   # Needs to remain an array at all times
        self.s_out.send(self.decision)

    def _pause(self) -> None:
        """Pause was called by the runtime"""
        super()._pause()

    def _stop(self) -> None:
        """Stop was called by the runtime"""
        super()._stop()
