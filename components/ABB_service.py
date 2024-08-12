from argparse import ArgumentParser
from Pyro5.api import expose, Daemon, locate_ns, oneway
from core.sensor.tactile_sensor_neuro import NeuroTac
import subprocess
import time

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import ABBController

@expose
class ABBService(object):
    def __init__(self, ip="192.168.125.1") -> None:
        self.robot = AsyncRobot(SyncRobot(ABBController(ip)))
        print(f"Connected to ABB")

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb) -> None:
        self.close()

    def close(self):
        self.robot.close()

    @property
    def info(self):
        return self.robot.info

    @property
    def tcp(self):
        return self.robot.tcp

    @tcp.setter
    def tcp(self, tcp):
        self.robot.tcp(tcp)

    @property
    def linear_speed(self):
        return self.robot.linear_speed

    @linear_speed.setter
    def linear_speed(self, speed):
        self.robot.linear_speed = speed

    @property
    def anglular_speed(self):
        return self.robot.anglular_speed

    @anglular_speed.setter
    def anglular_speed(self, speed):
        self.robot.anglular_speed = speed

    @property
    def blend_radius(self):
        return self.robot.blend_radius

    @blend_radius.setter
    def blend_radius(self, blend_radius):
        self.robot.blend_radius(blend_radius)

    @property
    def joint_angles(self):
        return self.robot.joint_angles

    @property
    def commanded_joint_angles(self):
        return self.robot.commanded_joint_angles

    @property
    def pose(self):
        return self.robot.pose

    @property
    def commanded_pose(self):
        return self.robot.command_pose

    @property
    def elbow(self):
        return self.robot.elbow

    @property
    def command_elbow(self):
        return self.robot.command_elbow

    @property
    def coord_frame(self):
        return self.robot.coord_frame

    @coord_frame.setter
    def coord_frame(self, frame):
        self.robot.coord_frame(frame)

    @oneway
    def move_joints(self, joint_angles):
        self.robot.move_joints(joint_angles)

    def move_joints_blocking(self, joint_angles):
        self.robot.move_joints(joint_angles)

    @oneway
    def move_linear(self, pose, elbow=None):
        self.robot.move_linear(self, pose, elbow=None)

    def move_linear_blocking(self, pose, elbow=None):
        self.robot.move_linear(self, pose, elbow=None)

    @oneway
    def move_circular(self, via_pose, end_pose, elbow=None):
        self.robot.move_circular(via_pose, end_pose, elbow=None)

    def move_circular_blocking(self, via_pose, end_pose, elbow=None):
        self.robot.move_circular(via_pose, end_pose, elbow=None)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--robot-ip", type=str, default="192.168.125.1", help="robot IP address"
    )
    parser.add_argument(
        "-i", "--host-ip", type=str, default="127.0.0.1", help="host IP address"
    )
    parser.add_argument(
        "-p", "--host-port", type=int, default=9092, help="host port number"
    )
    parser.add_argument(
        "-n",
        "--service-name",
        type=str,
        default="abb_service_1",
        help="service name",
    )
    args = vars(parser.parse_args())
    robot_ip = args["robot_ip"]
    service_name = args["service_name"]
    host_ip = args["host_ip"]
    host_port = args["host_port"]

    # Run background bash script that opens pyro namespace
    # This process is closed upon script exit
    print("Starting namespace server...")
    subprocess.Popen(["pyro5-ns"])
    time.sleep(2)

    with Daemon(host=host_ip, port=host_port) as daemon, locate_ns() as ns:
        print(f"Starting service {service_name} ...")
        service = ABBService()
        service_uri = daemon.register(service)
        ns.register(service_name, service_uri)
        print(f"Service {service_name} running (press CTRL-C to terminate) ...")
        daemon.requestLoop()

if __name__ == "__main__":
    main()
