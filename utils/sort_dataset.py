import shutil
import os
import pandas as pd

def main():
    dataset_path = "/media/farscope2/T7 Shield/Neuromorphic Data/Ao/bigdataset/"
    output_path = "/media/farscope2/T7 Shield/Neuromorphic Data/George/all_speeds_sorted/"
    try:
        os.mkdir(output_path)
    except:
        print("Directory exists already")

    # NOTE:
    # Data is currently saved in format:
    # ".bigdataset/TEXTUREA_TEXTUREB/...depth_X_speed_Y/events/...trial_Z_pose_T_events_on"

    # Find all textures within this path
    tex_dirs = os.listdir(dataset_path)
    textures = sum([d.split("_") for d in tex_dirs],[])
    num_textures = len(textures)

    # Find all speeds within this texture
    # NOTE: Assumes same speeds are collected for each
    speed_dirs = os.listdir(dataset_path+tex_dirs[0])
    speeds = list(set([s.split("_")[-1] for s in speed_dirs])) # Set to list to remove duplicates
    speeds = sorted([int(s) for s in speeds])   # Sort these in accending speed - requires conversion to int
    speeds = [str(s) for s in speeds]
    num_speeds = len(speeds)

    # Find all depths
    depth_dirs = os.listdir(dataset_path + tex_dirs[0])
    depths = list(set([d.split("_")[-3] for d in depth_dirs]))
    num_depths = len(depths)
    set_depth = 1.5

    print("Textures:")
    print(textures)
    print("Speeds:")
    print(speeds)   
    print("Depths:")
    print(depths)
    print(f"Set Depth: {set_depth}mm")

    # NOTE: This only takes data from set depth
    # Copy files into sorted folder
    for d in tex_dirs:
        # Find speed directories for these textures
        speed_d = os.listdir(dataset_path + d)

        dir_textures = d.split("_")

        for s in speed_d:
            path = f"{dataset_path}{d}/{s}/events/"
            files = os.listdir(path)

            # Set speed of each of these samples
            speed = s.split("_")[-1]
            depth = s.split("_")[-3]

            if depth == str(set_depth):
                for f in files:
                    # Textures are the directory name as pose 1 and 2
                    # Eg. MESH_FELT: pose_0 = MESH, pose_1 = FELT

                    split_name = f.split("_")
                    trial = split_name[-5]
                    pose = int(split_name[-3])

                    # Find texture and its position in the overall texture list
                    tex = dir_textures[pose]
                    texture = textures.index(tex)

                    # Copy file into sorted folder with new name
                    output_name = f"{trial}-{texture}-{speed}.pickle"
                    shutil.copyfile(path+f, output_path+output_name)

    print("Copied dataset")

    meta = pd.DataFrame.from_dict(
        data={
            "Num Textures": num_textures,
            "Num Speeds": num_speeds,
            "Textures": textures,
            "Speeds": speeds,
            "Depth": set_depth,
        },
        orient="index",
    )
    meta.to_csv(f"{output_path}/meta.csv")


if __name__ == "__main__":
    main()
