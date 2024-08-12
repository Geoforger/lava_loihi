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

# from cri.robot import SyncRobot, AsyncRobot
# from cri.controller import ABBController
from Pyro5.api import Proxy

class ABBController(AbstractProcess):

    def __init__(
        self,
        lookup_path: str,
        textures: list,
        speeds: np.ndarray,
        abb_params: dict,
        abb_service: str = "abb_service_1",
    ) -> None:

        self.lookup_path = lookup_path
        self.textures = textures
        self.speeds = speeds

        self.tcp = abb_params["tcp"]
        self.base_frame = abb_params["base_frame"]
        self.home_pose = abb_params["home_pose"]
        self.work_frame = abb_params["work_frame"]
        self.poses = abb_params["poses"]
        self.abb_service = abb_service

        super().__init__(
            lookup_path=self.lookup_path,
            textures=self.textures,
            speeds=self.speeds,
            tcp=self.tcp,
            base_frame=self.base_frame,
            home_pose=self.home_pose,
            work_frame=self.work_frame,
            poses=self.poses,
            abb_service=self.abb_service,
        )


@tag("fixed_pt")
@implements(proc=ABBController, protocol=LoihiProtocol)
@requires(CPU)
class PyABBController(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_SPARSE, int)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.lookup_path = proc_params["lookup_path"]
        self.textures = proc_params["textures"]
        self.speeds = proc_params["speeds"]

        self.tcp = proc_params["tcp"]
        self.base_frame = proc_params["base_frame"]
        self.home_pose = proc_params["home_pose"]
        self.work_frame = proc_params["work_frame"]
        self.poses = proc_params["poses"]
        self.abb_service = proc_params["abb_service"]

        # Init state machine lookup table
        self.lookup_table = np.load(self.lookup_path)

        # Move robot where needed
        self.robot = self.__make_pyro(self.abb_service)
        self.__robot_home()
        self.__robot_workframe()

    def run_spk(self) -> None:
        pass

    def run_post_mgmt(self):
        pass

    def __make_pyro(self, service) -> Proxy:
        """
        Private method that creates a connection to a specfied pyro5 server
        """
        return Proxy(f"PYRONAME:{service}")

    def __robot_home(self, linear_speed:int=50) -> None:
        # Set robot to base position
        print("Moving to home position ...")
        self.robot.coord_frame = self.base_frame
        self.robot.linear_speed = linear_speed
        self.robot.move_linear(self.home_pose)

        print("Robot at home position...")

    def __robot_workframe(self, linear_speed: int = 50) -> None:
        print("Moving to origin of work frame ...")
        self.robot.linear_speed = linear_speed
        self.robot.coord_frame = self.work_frame
        self.robot.move_linear((0, 0, 0, 0, 0, 0))
        print("Robot at work place origin...")

    def __robot_collect(self, pose:int) -> None:
        self.robot.move_linear(tap_move[0])
        self.robot.move_linear(tap_move[1])
        self.robot.move_linear(tap_move[2])

    def __state_change(self, idx_1, idx_2) -> int:
        speed_idx = np.argmax(self.lookup_table[idx_1, idx_2, :])
        return self.speeds[speed_idx]
