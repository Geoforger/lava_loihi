from plotting_functions import heatmap_lava, raster_plot
from data_processor import DataProcessor
import lava.lib.dl.slayer as slayer
import glob
import torch
import pandas as pd
from ast import literal_eval

def main():
    DATASET_PATH = "/media/george/T7 Shield/Neuromorphic Data/George/preproc_dataset_offset/"
    samples = glob.glob(f"{DATASET_PATH}*.npy")
    sample_path = samples[1]
    
    # Read meta for x, y sizes of preproc data
    meta = pd.read_csv(f"{DATASET_PATH}/meta.csv")
    meta['output_shape'] = meta['output_shape'].apply(literal_eval)
    x_size, y_size = meta["output_shape"].iloc[0]
    
    # event = slayer.io.read_np_spikes(sample_path)
    # # event.show()
    # spike = event.fill_tensor(
    #         torch.zeros(
    #             1, x_size, y_size, 1000, requires_grad=False
    #         ),
    #         sampling_time=1
    #     )
    
    # spike = spike.squeeze()
    
    # print(spike)
    # print(spike.shape)
        
    raster_plot(spike, display=True)

if __name__ == "__main__":
    main()