import shutil
import os
import pandas as pd

def main():
    dataset_path = "/media/farscope2/T7 Shield/Neuromorphic Data/Ao/0618_Slide_allspeed/"
    output_path = "/media/farscope2/T7 Shield/Neuromorphic Data/George/all_speeds_sorted/"
    try:
        os.mkdir(output_path)
    except:
        print("Directory exists already")

    # Find all textures within this path
    dir_list = os.listdir(dataset_path)
    textures = sum([d.split("_") for d in dir_list],[])
    num_textures = len(textures)

    # Find all speeds within this texture
    # NOTE: Assumes same speeds are collected for each
    speed_dirs = os.listdir(dataset_path+dir_list[0])
    speeds = [s.split("_")[-1] for s in speed_dirs]
    num_speeds = len(speeds)

    print("Textures:")
    print(textures)
    print("Speeds:")
    print(speeds)    

    # Copy files into sorted folder
    for d in dir_list:
        # Find speed directories for these textures
        speed_d = os.listdir(dataset_path + d)

        dir_textures = d.split("_")

        for s in speed_d:
            path = f"{dataset_path}{d}/{s}/events/"
            files = os.listdir(path)

            # Set speed of each of these samples
            speed = s.split("_")[-1]

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
                output_name = f"{trial}-{texture}-{speed}"
                shutil.copyfile(path+f, output_path+output_name)

    print("Copied dataset")

    meta = pd.DataFrame.from_dict(
        data={
            "Num Textures": num_textures,
            "Num Speeds": num_speeds,
            "Textures": textures,
            "Speeds": speeds,
            "Format": "trial-texture-speed",
        },
        orient="index",
    )
    meta.to_csv(f"{output_path}/meta.csv")


if __name__ == "__main__":
    main()
