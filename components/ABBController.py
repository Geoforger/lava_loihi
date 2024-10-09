import numpy as np
import time

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.process.variable import Var

from Pyro5.api import Proxy

class ABBController(AbstractProcess):
    def __init__(
        self,
        net_out_shape: tuple,
        lookup_path: str,
        forces: np.ndarray,
        speeds: np.ndarray,
        abb_params: dict,
        target_texture: dict,
        tex_index: int=0,
        abb_service: str = "abb_service_1",
        timeout: int = 5
    ) -> None:

        self.net_out_shape = net_out_shape
        self.lookup_path = lookup_path
        self.forces = forces
        self.speeds = speeds
        self.target_texture = target_texture
        self.abb_service = abb_service
        self.timeout = timeout

        self.robot_tcp = abb_params["robot_tcp"]
        self.base_frame = abb_params["base_frame"]
        self.home_pose = abb_params["home_pose"]
        self.work_frame = abb_params["work_frame"]
        self.tap_length = abb_params["tap_length"]

        self.acc_in = RefPort(self.net_out_shape)
        self.attempt = Var(shape=(1,), init=1)
        self.moving = Var(shape=(1,), init=False)

        # Start speed is speed that gives largest mean distances
        self.lookup_table = np.load(self.lookup_path)
        mean_dist = np.mean(self.lookup_table, axis=(0, 1))
        speed_idx = np.argmax(mean_dist)
        self.slide_speed = Var(shape=(1,), init=self.speeds[speed_idx])

        # Currently choose 0 or 1 of two different poses
        self.texture_position = [[0, 0, 0, 0, 0, 0], [97, 0, 0, 0, 0, 0]][tex_index]

        super().__init__(
            net_out_shape=self.net_out_shape,
            lookup_path=self.lookup_path,
            forces=self.forces,
            speeds=self.speeds,
            texture_position=self.texture_position,
            robot_tcp=self.robot_tcp,
            base_frame=self.base_frame,
            home_pose=self.home_pose,
            work_frame=self.work_frame,
            tap_length=self.tap_length,
            abb_service=self.abb_service,
            target_texture=self.target_texture,
            timeout=self.timeout,
        )


@tag("fixed_pt")
@implements(proc=ABBController, protocol=LoihiProtocol)
@requires(CPU)
class PyABBController(PyLoihiProcessModel):
    acc_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32)

    attempt: np.ndarray = LavaPyType(np.ndarray, int)
    slide_speed: np.ndarray = LavaPyType(np.ndarray, int)
    moving: np.ndarray = LavaPyType(np.ndarray, bool)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.lookup_path = proc_params["lookup_path"]
        self.forces = proc_params["forces"]
        self.speeds = proc_params["speeds"]
        self.target_texture = proc_params["target_texture"]
        self.net_out_shape = proc_params["net_out_shape"]
        self.timeout = proc_params["timeout"]
        self.texture_position = proc_params["texture_position"]

        self.robot_tcp = proc_params["robot_tcp"]
        self.base_frame = proc_params["base_frame"]
        self.home_pose = proc_params["home_pose"]
        self.work_frame = proc_params["work_frame"]
        self.tap_length = proc_params["tap_length"]
        self.abb_service = proc_params["abb_service"]

        # Init state machine lookup table
        self.lookup_table = np.load(self.lookup_path)

        # Pose and tap movements for target texture
        texture_x = self.texture_position[0]
        contact_z = self.work_frame[2] - self.forces[self.target_texture]
        self.tap_moves = (
            [texture_x, 0, contact_z, 0, 0, 0],
            [texture_x, self.tap_length, contact_z, 0, 0, 0],
            [texture_x, self.tap_length, 0, 0, 0, 0],
        )

        # Flags to control workflow
        self.moving = np.array([False])
        self.attempt = np.array([1])
        self.attempt_time_step = 0
        self.finished = False

        self.accumulator = np.zeros(self.net_out_shape)

        # Move robot where needed
        self.robot = self.__make_pyro(self.abb_service)
        print("Connected to ABB pyro...")
        self.robot.tcp = self.robot_tcp
        # self.__robot_home()
        self.__robot_workframe(workframe_origin=True)

    def run_spk(self) -> None:
        if not self.finished:
            self.moving = np.array([self.robot.moving])     # Flag stored in controller class - does not poll robot directly
            sample_ts = self.time_step - self.attempt_time_step

            if not self.moving[0]:
                # For first attempt
                # At first time step move to tap position
                if sample_ts == 1:
                    print(f"Starting attempt: {self.attempt}")
                    self.__intiate_tap()
                    # This copy prevent overwriting the self.slide_speed with an int value that cant be sent down ref port
                    s = self.slide_speed.copy()[0]
                    self.robot.linear_speed = int(s)

                # Tap and start slide
                elif sample_ts == 2:
                    print("Starting slide")
                    self.robot.move_linear(self.tap_moves[1])
                    print("Sliding")

                # If stopped moving and not in the first steps initiate again
                else:
                    self.robot.move_linear_blocking(self.tap_moves[2])
                    print(f"Reset ts: {self.time_step}")
                    self.__robot_workframe(workframe_origin=False)
                    self.slide_speed = self.__state_change()
                    self.attempt_time_step = self.time_step

                    if self.attempt >= self.timeout:
                        print("Max attempts reached")
                        self.finished = True
                    else:
                        print(f"Finished attempt: {self.attempt}")
                        self.attempt += 1

                    print("Not Timed out. Starting next attempt...")

    def __make_pyro(self, service) -> Proxy:
        """
        Private method that creates a connection to a specfied pyro5 server
        """
        return Proxy(f"PYRONAME:{service}")

    def __robot_home(self, linear_speed:int=80) -> None:
        # Set robot to base position
        print("Moving to home position ...")
        self.robot.coord_frame = self.base_frame
        self.robot.linear_speed = linear_speed
        self.robot.move_linear_blocking(self.home_pose)
        print("Robot at home position...")

    def __robot_workframe(self, linear_speed: int = 80, workframe_origin:bool = False) -> None:
        self.robot.coord_frame = self.work_frame
        self.robot.linear_speed = linear_speed
        if workframe_origin:
            print("Moving to origin of work frame ...")
            self.robot.move_linear_blocking([0, 0, 0, 0, 0, 0])
            print("Robot at work place origin...")
        print("Moving to texture pose...")
        self.robot.move_linear_blocking(self.texture_position)
        print("ABB ready for test.")

    def __intiate_tap(self) -> None:
        self.robot.coord_frame = self.work_frame
        print("Moving to obj pose...")
        self.robot.move_linear_blocking(self.texture_position)
        print("Initiating contact...")
        self.robot.linear_speed = 10
        self.robot.move_linear_blocking(self.tap_moves[0])
        print("Sensor in contact")

    def __state_change(self) -> np.ndarray:
        arg_sort = np.argsort(self.accumulator)
        highest = arg_sort[-1]
        second = arg_sort[-2]

        # NOTE: This will be replaced when using correct network
        if highest > 9:
            if second != 9:
                highest = second + 1
            elif second != 0:
                highest = second - 1

        if second > 9:
            if highest != 9:
                second = highest + 1
            elif highest != 0:
                second = highest - 1

        speed_idx = np.argmax(self.lookup_table[highest, second, :])
        return np.array([self.speeds[speed_idx]])

    def post_guard(self) -> bool:
        return True

    def run_post_mgmt(self) -> None:
        self.accumulator += self.acc_in.read()

    def _pause(self) -> None:
        """Pause was called by the runtime"""
        super()._pause()

    def _stop(self) -> None:
        """Stop was called by the runtime"""
        if self.robot is not None:
            self.robot._pyroRelease()   # Release the connection to the proxy service
        super()._stop()
