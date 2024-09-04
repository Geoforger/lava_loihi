import numpy as np
import threading

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import ABBController
from core.sensor.tactile_sensor_neuro import NeuroTac


def make_robot():
    return AsyncRobot(SyncRobot(ABBController(ip="192.168.125.1")))

def make_sensor():
    return NeuroTac(save_events_video=False, save_acc_video=False, display=False)

# def make_meta(
#         depth,
#         robot_tcp = [0, 0, 59, 0, 0, 0], # Size of the TacTip (tcp = tool center point)
#         base_frame = [0, 0, 0, 0, 0, 0], # Origin of the base frame is at centre of the ABB robot base0
#         home_pose = [400, 0, 240, 180, 0, 180], # Starting point of the robot when switched on
#         work_frame = [465, -200, 26, 180, 0, 180], # Starting point and reference frame for this experiment
#         linear_speed = 20, # Robot's speed for linear movements default = 10
#         angular_speed = 20, # Robot's speed for rotation movements
#         tap_move = None,   # Sequence of positions for the tap relative to the robot's current pose (positive z goes down)
#         # obj_poses = [[0, y, 0, 0, 0, 0]for y in range(0, 151, 15)], # Position of the different objects/poses relative to the work frame origin
#         obj_poses = [[0, 0, 0, 0, 0, 0], [102, 0, 2.2, 0, 0, 0]], # -10: up, 10: down
#         n_trials = 100, # Number of taps at each pose
#     ):

#     if tap_move is None:
#         tap_move = [[0, 0, 3.8 + depth, 0, 0, 0], [0, 30, 3.8 + depth, 0, 0, 0], [0, 30, 0, 0, 0, 0]]
#     meta = locals().copy()
#     return meta

def collect_label(label_idx, position_idx, speeds, depths, output_path, tap_distance=60):
    robot_tcp = [0, 0, 59, 0, 0, 0] # Size of the TacTip (tcp = tool center point)
    base_frame = [0, 0, 0, 0, 0, 0] # Origin of the base frame is at centre of the ABB robot base0
    home_pose = [400, 0, 240, 180, 0, 180] # Starting point of the robot when switched on
    work_frame = [465, -200, 26, 180, 0, 180] # Starting point and reference frame for this experiment 
    obj_poses = [[0, 0, 0, 0, 0, 0], [102, 0, 0, 0, 0, 0]]
    n_trials = 100

    # Select work frame position between two options
    start_pose = obj_poses[position_idx]

    # Select depth coordinates
    depths = [depths[force][label_idx] for force in depths.key()]

    with make_robot() as robot, make_sensor() as sensor:
        robot.tcp = robot_tcp

        # move to home position
        print("Moving to home position ...")
        robot.coord_frame = base_frame
        robot.linear_speed = 80
        robot.angular_speed = 20
        robot.move_linear(home_pose)
        print("Arm at home pose")

        # Move to origin of work frame
        print("Moving to origin of work frame ...")
        robot.coord_frame = work_frame
        robot.move_linear((0, 0, 0, 0, 0, 0))
        print("At origin of work frame")

        # For the given texture, iterate over depth, speed and trials
        for depth_idx, depth in enumerate(depths):

            # Calculate tap distance for this texture and force
            depth_z = work_frame[2] - depth

            # TODO: Does this require an offset?
            # Apply tap move settings
            tap_move = [
                [0, 0, depth_z, 0, 0, 0],  # Tap down
                [0, tap_distance, depth_z, 0, 0, 0],  # Slide
                [0, tap_distance, 0, 0, 0, 0],  # Tap up
            ]

            # Get newton value for force for labels
            force = list(depths.keys())[depth_idx]

            for speed in speeds:
                for trial_idx in range(n_trials):
                    # Set output file names
                    events_on_file = f"{output_path}/{force}-{speed}-{label_idx}-{trial_idx}_on.pickle"
                    events_off_file = f"{output_path}/{force}-{speed}-{label_idx}-{trial_idx}_off.pickle"
                    sensor.set_filenames(events_on_file=events_on_file, events_off_file=events_off_file)

                    sensor.reset_variables()

                    print("Moving to start pose")
                    robot.linear_speed = 80
                    robot.move_linear(start_pose)
                    print("At start position")

                    # Tap
                    robot.coord_frame = base_frame
                    robot.coord_frame = robot.pose

                    #  Initiate tap
                    robot.linear_speed = 40
                    robot.move_linear(tap_move[0])

                    # Set to target sliding speed in mm/s
                    robot.linear_speed = speed

                    #  Start sensor recording when sliding
                    sensor.start_logging()
                    t = threading.Thread(target=sensor.get_events, args=())
                    t.start()

                    # Slide - Blocking movement
                    robot.move_linear(tap_move[1])

                    # Stop sensor recording after sliding
                    sensor.stop_logging()
                    t.join()

                    # Lift sensor
                    robot.move_linear(tap_move[2])

                    # Collate proper timestamp values in ms.
                    sensor.value_cleanup()

                    # Save data
                    sensor.save_events_on(events_on_file)
                    sensor.save_events_off(events_off_file)
                    print("saved data")

        # Move to home position
        print("Finished collection...")
        print("Moving to home position ...")
        robot.coord_frame = base_frame
        robot.linear_speed = 80
        robot.move_linear(home_pose)
        print("Robot at home position")


def main():
    output_path = ""
    collect_textures = ["Mesh", "Felt"]   # Change whenever changing physical textures
    speeds = np.arange(15, 65, 10)
    forces = ["1N", "1.5N", "2.5N"]

    texture_labels = {
        "Mesh": 0,
        "Felt": 1,
        "Cotton": 2,
        "Nylon": 3,
        "Fur": 4,
        "Wood": 5,
        "Acrylic": 6,
        "FashionFabric": 7,
        "Wool": 8,
        "Canvas": 9
    }

    force_depths = {
        "1N": [],
        "1.5N": [],
        "2N": []
    }

    # Collect each tex at each depth in turn
    for pos_idx, tex in enumerate(range(collect_textures)):
        print(f"Starting collection for texture: {tex}")
        # Find the tex index
        tex_idx = texture_labels[tex]
        collect_label(tex_idx, pos_idx, speeds, force_depths, output_path)


if __name__ == "__main__":
    main()
