import numpy as np
import pandas as pd
import glob

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.variable import Var


class Datalogger(AbstractProcess):
    def __init__(
        self,
        out_path:str,
        target_label:int,
    ) -> None:

        self.out_path = out_path
        self.target_label = target_label

        # What data do we want to record
        self.conf_in = RefPort((1,))
        self.decision_in = RefPort((1,))
        self.arm_speed_in = RefPort((1,))

        super().__init__(
            out_path = self.out_path,
            target_label = self.target_label,
            test_num = self.test_num
        )


@tag("fixed_pt")
@implements(proc=Datalogger, protocol=LoihiProtocol)
@requires(CPU)
class PyDatalogger(PyLoihiProcessModel):
    conf_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)
    decision_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int8)
    arm_speed_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int8)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.out_path = proc_params["out_path"]
        self.target_label = proc_params["target_label"]

        self.conf = 0.0
        self.decision = None
        self.arm_speed = None

        # Find the number of files in the output dir
        self.test_num = len(glob.glob(f"{self.out_path}/{self.target_label}-iteration-*.csv"))

        # Create output dataframe
        self.data = pd.DataFrame(
            columns=["Time Step", "Arm Speed", "Target Label", "Decision", "Confidence"]
        )

    def run_spk(self) -> None:
        # Append current values to csv file
        inp = {
            "Time Step": self.time_step,
            "Arm Speed": self.arm_speed,
            "Target Label": self.target_label,
            "Decision": self.decision,
            "Confidence": self.conf
        }

        self.data = pd.concat([self.data, pd.DataFrame(inp, index=[0])], ignore_index=True)

        self.data.to_csv(
            f"{self.out_path}/{self.target_label}-iteration-{self.test_num}.csv"
        )

    def run_post_mgmt(self):
        # Recieve values from ref ports
        self.conf = self.conf_in.read()
        self.decision = self.decision_in.read()
        self.arm_speed = self.arm_speed_in.read()

    def post_guard(self):
        return True

    def _pause(self) -> None:
        """Pause was called by the runtime"""
        super()._pause()

    def _stop(self) -> None:
        """Stop was called by the runtime"""
        super()._stop()
