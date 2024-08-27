import numpy as np

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
        textures: dict,
        speeds: np.ndarray,
        abb_params: dict,
        target_texture: dict,
        abb_service: str = "abb_service_1",
        timeout: int = 5
    ) -> None:

        self.net_out_shape = net_out_shape
        self.lookup_path = lookup_path
        self.textures = textures
        self.speeds = speeds
        self.target_texture = target_texture
        self.abb_service = abb_service
        self.timeout = timeout

        self.robot_tcp = abb_params["robot_tcp"]
        self.base_frame = abb_params["base_frame"]
        self.home_pose = abb_params["home_pose"]
        self.work_frame = abb_params["work_frame"]
        self.tap_depth = abb_params["tap_depth"]
        self.tap_length = abb_params["tap_length"]

        self.acc_in = RefPort(self.net_out_shape)
        self.slide_speed = Var(shape=(1,), init=0)

        super().__init__(
            net_out_shape=self.net_out_shape,
            lookup_path=self.lookup_path,
            textures=self.textures,
            speeds=self.speeds,
            robot_tcp=self.robot_tcp,
            base_frame=self.base_frame,
            home_pose=self.home_pose,
            work_frame=self.work_frame,
            tap_depth=self.tap_depth,
            tap_length=self.tap_length,
            abb_service=self.abb_service,
            target_texture=self.target_texture,
            timeout=self.timeout
        )


@tag("fixed_pt")
@implements(proc=ABBController, protocol=LoihiProtocol)
@requires(CPU)
class PyABBController(PyLoihiProcessModel):
    acc_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.lookup_path = proc_params["lookup_path"]
        self.textures = proc_params["textures"]
        self.speeds = proc_params["speeds"]
        self.target_texture = proc_params["target_texture"]
        self.net_out_shape = proc_params["net_out_shape"]
        self.timeout = proc_params["timeout"]

        self.robot_tcp = proc_params["robot_tcp"]
        self.base_frame = proc_params["base_frame"]
        self.home_pose = proc_params["home_pose"]
        self.work_frame = proc_params["work_frame"]
        self.tap_depth = proc_params["tap_depth"]
        self.tap_length = proc_params["tap_length"]
        self.abb_service = proc_params["abb_service"]

        # Init state machine lookup table
        self.lookup_table = np.load(self.lookup_path)

        # Pose and tap movements for target texture
        # NOTE: Only 1 texture per simulation currently
        contact_z = self.work_frame[2] - self.textures[self.target_texture]
        self.obj_pose = [0, 0, contact_z, 0, 0, 0]
        self.tap_moves = (
                [0, 0, contact_z + self.tap_depth, 0, 0, 0],
                [0, self.tap_length, contact_z + self.tap_depth, 0, 0, 0],
                [0, self.tap_length, 0, 0, 0, 0],
        )

        # Start speed is speed that gives largest mean distances
        mean_dist = np.mean(self.lookup_table, axis=(0,1))
        speed_idx = np.argmax(mean_dist)
        self.slide_speed = int(self.speeds[speed_idx])

        # Flags to control workflow
        self.moving = False
        self.attempts = 0
        self.attempt_time_step = 0
        self.finished = np.array([0])

        self.accumulator = np.zeros(self.net_out_shape)

        # Move robot where needed
        self.robot = self.__make_pyro(self.abb_service)
        print("Connected to ABB pyro...")
        self.robot.tcp = self.robot_tcp
        self.__robot_home()
        self.__robot_workframe()

    def run_spk(self) -> None:
        self.moving = self.robot.moving     # Flag stored in controller class - does not poll robot directly
        sample_ts = self.time_step - self.attempt_time_step

        if not self.moving:
            # For first attempt
            # At first time step move to tap position
            if sample_ts == 1:
                print(f"Starting attempt: {self.attempts}")
                self.__intiate_tap()
                self.robot.linear_speed = self.slide_speed

            # Tap and start slide
            elif sample_ts == 2:
                self.robot.move_linear(self.tap_moves[1])
                print("Starting slide")
                self.robot.move_linear(self.tap_moves[2])
                print("Sliding")

            # If stopped moving and not in the first steps initiate again
            else:
                self._reset_tap()
                self.__state_change()
                self.attempt_time_step = self.time_step
                self.attempts += 1

                if self.attempts > self.timeout:
                    self._stop()

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
        self.robot.move_linear_blocking(self.home_pose)
        print("Robot at home position...")

    def __robot_workframe(self, linear_speed: int = 50) -> None:
        print("Moving to origin of work frame ...")
        self.robot.coord_frame = self.work_frame
        self.robot.linear_speed = linear_speed
        self.robot.move_linear_blocking([0, 0, 0, 0, 0, 0])
        print("Robot at work place origin...")
        print("ABB ready for test...")

    def __intiate_tap(self) -> None:
        self.robot.coord_frame = self.work_frame
        print("Moving to obj pose...")
        self.robot.move_linear_blocking(self.obj_pose)
        print("Initiating contact...")
        self.robot.linear_speed = 10
        self.robot.move_linear_blocking(self.tap_moves[0])
        print("Sensor in contact")

    def __state_change(self) -> int:
        arg_sort = np.argsort(self.accumulator)
        highest = arg_sort[-1]
        second = arg_sort[-2]

        speed_idx = np.argmax(self.lookup_table[highest, second, :])
        return self.speeds[speed_idx]

    def _reset_tap(self):
        print("Moving to workframe...")
        self.__robot_workframe()
        print("Robot at workframe")
        self.__intiate_tap()

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        self.accumulator += self.acc_in.read()

    def _pause(self) -> None:
        """Pause was called by the runtime"""
        super()._pause()

    def _stop(self) -> None:
        """Stop was called by the runtime"""
        if self.robot is not None:  # TODO: Implement if not robot connection
            self.robot.close()
        super()._stop()
