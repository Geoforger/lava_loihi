import numpy as np
import pandas as pd
import glob
import scipy.stats as stats

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyRefPort, PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import RefPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.variable import Var


class Datalogger(AbstractProcess):
    def __init__(
        self,
        out_path:str,
        net_out_shape:tuple,
        target_label:int,
    ) -> None:

        self.out_path = out_path
        self.target_label = target_label
        self.net_out_shape = net_out_shape

        # What data do we want to record
        self.decision_in = InPort((1,))
        self.conf_in = RefPort((1,))
        self.arm_speed_in = RefPort((1,))
        self.attempt_in = RefPort((1,))
        self.acc_in = RefPort(self.net_out_shape)

        super().__init__(
            out_path = self.out_path,
            net_out_shape=self.net_out_shape,
            target_label = self.target_label,
        )


@tag("fixed_pt")
@implements(proc=Datalogger, protocol=LoihiProtocol)
@requires(CPU)
class PyDatalogger(PyLoihiProcessModel):
    decision_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int8)
    conf_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    arm_speed_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
    attempt_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, int)
    acc_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int64)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.out_path = proc_params["out_path"]
        self.target_label = proc_params["target_label"]
        self.net_out_shape = proc_params["net_out_shape"]

        self.conf = 0.0
        self.decision = None
        self.arm_speed = None
        self.attempt = None
        self.accumulator = np.zeros(self.net_out_shape)

        # Find the number of files in the output dir
        self.test_num = len(glob.glob(f"{self.out_path}/{self.target_label}-iteration-*.csv"))

        # Create output dataframe
        self.data = pd.DataFrame(
            columns=[
                "Time Step",
                "Arm Speed",
                "Target Label",
                "Decision",
                "Confidence",
                "Entropy",
                "Num Spikes",
                "Attempt",
            ]
        )

    def run_spk(self) -> None:
        self.decision = self.decision_in.recv()

        # Avoids divide by zero error if no spikes yet
        if np.sum(self.accumulator) > 0:
            entropy = stats.entropy(self.accumulator, base=2)
        else:
            entropy = 0.0

        # Append current values to csv file
        inp = {
            "Time Step": self.time_step,
            "Arm Speed": self.arm_speed,
            "Target Label": self.target_label,
            "Decision": self.decision,
            "Confidence": self.conf,
            "Entropy": entropy,
            "Num Spikes": np.sum(self.accumulator),
            "Attempt": self.attempt,
        }

        self.data = pd.concat([self.data, pd.DataFrame(inp, index=[0])], ignore_index=True)

        self.data.to_csv(
            f"{self.out_path}/{self.target_label}-iteration-{self.test_num}.csv"
        )

    def run_post_mgmt(self):
        # Recieve values from ref ports
        self.conf = self.conf_in.read()
        self.arm_speed = self.arm_speed_in.read()
        self.attempt = self.attempt_in.read()
        self.accumulator += self.acc_in.read()

    def post_guard(self):
        return True

    def _pause(self) -> None:
        """Pause was called by the runtime"""
        super()._pause()

    def _stop(self) -> None:
        """Stop was called by the runtime"""
        super()._stop()
