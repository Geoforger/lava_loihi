import numpy as np
import glob
import os
import pandas as pd
import lava.lib.dl.slayer as slayer
import concurrent.futures
from utils import nums_from_string
from data_processor import DataProcessor


def PreprocessSample(**kwargs):
    """
    Function to preprocess a single sample from dataset. Sample will be saved with same filename as input

    Parameters:
        **kwargs: Arguments for preprocessing:
            OUTPUT_PATH (str): Output path string to save directory
            sample (String): String path with filename
            lava (bool): Bool to indicate whether to output lava event object. Default False
            offset (int): Integer to offset the sample times by (in ms)
            cuttoff (int): Integer to set the upper limit of events in the sample (in ms)
            Rmv_Duplicates (bool): Bool to indicate if duplicate events should be removed from sample
            pixel_reduction (list): List of pixels to removed from each dimension of camera frame. Format: [x_left, x_right, y_top, y_bot]
            save (bool): Bool to indicate whether to save processed data or simply return from function
    Returns:
        sample (numpy array or lava event object): Preprocessed sample as either a numpy array or lava event object
    """
    # input_path = kwargs["DATASET_PATH"]
    filename = kwargs["sample"]
    try:
        data = DataProcessor.load_data(path=filename)

        # Check args presented and apply preprocessing
        if "pixel_reduction" in kwargs:
            pixel_vals = kwargs["pixel_reduction"]
            data.pixel_reduction(pixel_vals[0], pixel_vals[1], pixel_vals[2], pixel_vals[3])

        # Clip to start of sample
        if "start_thresh" in kwargs:
            sample_start = data.find_start(threshold=kwargs["start_thresh"])
            data.offset_values(sample_start, reduce=True)

        if "offset" in kwargs:
            data.offset_values(kwargs["offset"], reduce=True)
        if "cuttoff" in kwargs:
            data.remove_cuttoff(kwargs["cuttoff"])
        if "rmv_duplicates" in kwargs:
            data.remove_duplicates()
        if "pooling" in kwargs:
            kernel = kwargs["kernel"]
            stride = kwargs["stride"]
            threshold = kwargs["threshold"]
            data.threshold_pooling(kernel, stride, threshold)
        output_shape = data.data.shape

        if "save" in kwargs and "OUTPUT_PATH" in kwargs:
            out_path = kwargs["OUTPUT_PATH"]
            filename = f"{filename.split('/')[-1]}"

            if "lava" in kwargs and kwargs["lava"] is True:
                data.create_events()
                slayer.io.encode_np_spikes(f"{out_path}/{filename}", data.data)
            else:
                data.save_data_np(f"{out_path}/{filename}")

        elif "save" in kwargs and "OUTPUT_PATH" not in kwargs:
            raise Exception("No filename provided to save processed data to")

        print("Processed data")
        return data.data, output_shape
    
    except Exception as e:
        # Log sample name if fail to import
        d = {"filename":[filename]}
        print(e)
        print(d)
        return

def PreprocessDataset(**kwargs):
    """
    Function to preprocess dataset based on input parameters. Uses multiprocessing to reduce time taken.

    Parameters:
        kwargs: Arguments for preprocessing:
            DATASET_PATH (string): String of path to dataset requiring preprocessing
            OUTPUT_PATH (string): String of path to directory for preprocessed data
            lava (bool): Bool to indicate whether the input data is lava events. Default False
            offset (int): Integer to offset the sample times by (in ms)
            cuttoff (int): Integer to set the upper limit of events in the sample (in ms)
            rmv_duplicates (bool): Bool to indicate if duplicate events should be removed from sample
            pixel_reduction (list): List of pixels to removed from each dimension of camera frame. Format: [x_left, x_right, y_top, y_bot]
    Returns:
        None
    """
    args = kwargs.copy()
    args["save"] = True

    # Create output folder
    out_path = args["OUTPUT_PATH"]
    if os.path.isdir(out_path):
        d = input(f"Output path {out_path} exists. Overwrite? (Y/n)")
        if d == "n":
            raise Exception("Not overwriting output directory")
        else:
            print("Overwriting Directory")
    else:  
        os.mkdir(args["OUTPUT_PATH"])        

    path = args.get("DATASET_PATH", None)
    filenames = glob.glob(f"{path}/*.pickle") 

    arg_dicts = [args.copy() for _ in range(len(filenames))]
    for idx, file in enumerate(filenames):
        arg_dicts[idx].update({"sample":file})

    results = []
    # Pool multiprocessor using lambda function to unpack arg dictionary
    print("Starting process pool...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(PreprocessSample, **arg_dicts[file]) for file in range(len(filenames))]

        print("Getting results...")
        for f in concurrent.futures.as_completed(results):
            _, shape = f.result()

    # Save args to a meta file
    args["output_shape"] = shape
    print(args)
    # df = pd.DataFrame.from_dict(args, orient="columns")
    df = pd.DataFrame([args])
    df.to_csv(f"{out_path}/meta.csv")

    print("Finished processing dataset")


def main(): 
    dataset = "/media/george/T7 Shield/Neuromorphic Data/George/speed_depth_dataset"
    output = "/media/george/T7 Shield/Neuromorphic Data/George/preprocessed_new_dataset/"

    args = {
        "DATASET_PATH": dataset,
        "OUTPUT_PATH": output,
        "pixel_reduction": (195, 170, 102, 110),
        "lava": False,
        "save": True,
        # "start_thresh": 50,
        # "offset": 100,
        "cuttoff": 1000,
        "rmv_duplicates": True,
        "pooling": True,
        "kernel": (4, 4),
        "stride": 4,
        "threshold": 1,
    }

    PreprocessDataset(**args)


if __name__ == "__main__":
    main()
