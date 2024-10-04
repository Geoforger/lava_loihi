# -------------------------------------------------------------------------------------------------------------------------------------------------------
# Created By  : George Brayshaw
# Created Date: 03/May/2022
# version ='1.5'
# python version = '3.8.10'
# ------------------------------------------------------------------------------------------------------------------------------------------------------
"""This file provides a data processing class for manipulating neuroTac data

File contains 2 dependancies NOT included in a base Python 3.8.10 installation (numpy and nums_from_string).
Class has been tested on python 3.8.10 but should work on anything >3.6 (anyone willing to test this please report back).

TODO: Refactor messy code throughout
"""
# ------------------------------------------------------------------------------------------------------------------------------------------------------
import pickle
import numpy as np

# import os
# import nums_from_string     # pip install nums_from_string
import time

# import torch  # Used to create torch tensor for integration with lava
import lava.lib.dl.slayer as slayer  # Used for creating the Events object
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


class DataProcessor:
    """Class to process neuroTac data. Use the .load_data method to load from a previously saved file
    Arguments
    ---------
    data:   numpy array of lists
                neuroTac sample to process
    AER:    Bool (default=False)
                Bool to indicate if loaded data is in AER format
    """
    def __init__(self, data=None, AER=False):
        self.data = data
        # Flag to indicate if data has been converted to AER and thus made unavailable for most  operations
        self.__AER = AER

    @classmethod
    def load_data(cls, path: str, AER: bool = False):
        """Class method to load data in from a file

        path (string): Path string to the location of data if load_data = True

        """
        with open(path, "rb") as openfile:
            try:
                data = pickle.load(openfile)
            except EOFError:
                print(EOFError)

        return cls(data, AER=AER)

    @classmethod
    def load_data_np(cls, path: str, AER: bool = False):
        """Class method to load data in from a file

        path (string): Path string to the location of data if load_data = True

        """
        data = np.load(path, allow_pickle=True)

        return cls(data, AER=AER)

    def save_data(self, PATH):
        """Function to save processed data
        Arguments
        ---------
        PATH:       string
                        Path to save location for data. NOTE: Currently you must specify the filename and type in this PATH string
        """
        with open(PATH, "wb") as pickle_out:
            pickle.dump(self.data, pickle_out)
            pickle_out.close()

    def save_data_np(self, PATH):
        np.save(PATH, self.data, allow_pickle=True)

    def data_show(self):
        """ " Function to print the current data stored in the class"""
        print(self.data)

    def plot_data(self, display=False, normalise=False):
        """
        Method to plot the currently stored data as a heatmap
        """
        unproc_data = np.zeros(self.data.shape)
        for y in range(self.data.shape[0]):
            for x in range(self.data.shape[1]):
                for spike in list(self.data[y, x]):
                    unproc_data[y, x] += 1

        if normalise:
            ax = sns.heatmap(unproc_data, linewidth=0.0, norm=LogNorm())
        else:
            ax = sns.heatmap(unproc_data, linewidth=0.0)

        if display:
            plt.show()

        return ax

    def remove_empty(self):
        """ " Function to remove empty lists from input data"""
        if not self.__AER:
            new_list = np.empty_like(self.data)

            for row in range(self.data.shape[0]):
                for column in range(self.data.shape[1]):
                    # Use list(set()) to remove duplicates in list (sets cannot contain dupes)
                    if self.data[row, column] != []:
                        new_list[row, column] = self.data[row, column]

            self.data = new_list

        else:
            raise ValueError(
                "Data is in AER format and cannot perform remove_empty operation"
            )

    # Function is currently bugged
    def remove_duplicates(self):
        """Function to remove duplicate lists from input data"""
        if not self.__AER:
            new_list = np.empty_like(self.data)

            for row in range(self.data.shape[0]):
                for column in range(self.data.shape[1]):
                    # Use list(set()) to remove duplicates in list (sets cannot contain dupes)
                    new_list[row, column] = list(set(self.data[row, column]))

            self.data = new_list

        else:
            raise ValueError(
                "Data is in AER format and cannot perform remove_duplicates operation"
            )

    def remove_cuttoff(self, cuttoff, remove_dup=False, remove_em=False):
        """Function to remove events after a specified cuttoff point
        Arguments
        ---------
        cuttoff:        int
                            Cuttoff point after which all entries in nested list should be removed
        remove_dup:     Bool (default=False)
                            Bool to indicate if you wish to remove duplicate spikes before processing
        remove_em:      Bool (default=False)
                            Bool to indicate if you wish to remove empty pixels before processing
        """
        if not self.__AER:
            if remove_dup:
                self.remove_duplicates()
            if remove_em:
                self.remove_empty()

            new_list = np.empty_like(self.data)

            for row in range(self.data.shape[0]):
                for column in range(self.data.shape[1]):
                    # Create new list for each pixel, only containing values lower than cuttoff
                    new_list[row, column] = [
                        element
                        for element in self.data[row, column]
                        if element <= cuttoff
                    ]

            self.data = new_list

        else:
            raise ValueError(
                "Data is in AER format and cannot perform remove_cuttoff operation"
            )

    def pixel_reduction(self, x_reduce_l, x_reduce_r, y_reduce_t, y_reduce_b):
        """Function to reduce the number of pixels in output image from the neuroTac

        Arguments
        ----------
        x_reduce_l:   int
                        Total number of pixels to remove from the left of the array
        x_reduce_r:   int
                        Total number of pixels to remove from the right of the array
        y_reduce_t:   int
                        Total number of pixels to remove from the top of the array
        y_reduce_b:   int
                        Total number of pixels to remove from the bottom of the array
        Returns
        -------
        reduced_image:  nested list (array of lists)
                            New cropped array of timestamps
        """
        if self.__AER is False:
            # Find shape of input data
            y_size, x_size = self.data.shape

            # Find number of pixels to crop from right and bottom
            x_r = x_size - x_reduce_r
            y_b = y_size - y_reduce_b

            # Create new array as a slice of old array
            reduced_image = self.data[y_reduce_t:y_b, x_reduce_l:x_r]

            self.data = reduced_image
            # return reduced_image

        else:
            raise ValueError(
                "Data is in AER format and cannot perform pixel_reduction operation"
            )

    def __bitstring_to_bytes(self, s):
        """
        Private method used in the convert_to_aer function
        """
        v = int(s, 2)
        b = bytearray()
        while v:
            b.append(v & 0xFF)
            v >>= 8
        return bytes(b[::-1])

    def convert_to_aer(self, ON_OFF=1):
        """Function to create convert pickled neuroTac data into aer format .bin file

        Arguments
        ----------
        ON_OFF:     Integer
                        Input to state whether data contains OFF events. 0 = OFF events included. Default = 1
        """

        # Create a temporary list to contain information for all events
        temp_list = []

        # Cycle through each nested list and check if empty
        # Check row
        for y in range(self.data.shape[0]):
            # Check column
            for x in range(self.data.shape[1]):
                # Check if pixel (x,y) is empty
                if self.data[y, x]:
                    # Cycle through all events in this pixel
                    for spike in self.data[y, x]:
                        # Currently we only input ON events
                        temp_list.append([x, y, ON_OFF, spike])

        # Order by timestamp with earliest spike first
        sorted_list = sorted(temp_list, key=lambda x: int(x[3]), reverse=False)

        # Create a byte for entire datasample
        string = bytearray()

        # Convert to 40 bit binary number
        for element in sorted_list:
            # Add each of x,y,p & ts of the list to the byte array
            # print(f'element[0] = {element[0]}')
            if element[0] == 0:
                string.extend(bytearray([0]))
            else:
                string.extend(bytearray([element[0]]))

            # print(f'element[1] = {element[1]}')
            if element[1] == 0:
                string.extend(bytearray([0]))
            else:
                string.extend(bytearray([element[1]]))

            # print(f'element[2] = {element[2]}')
            if element[2] == 0:
                string.extend(bytearray([0]))
            else:
                string.extend(bytearray([element[2] << 7]))

            # Timestamp cannot be 0
            # If it's less than 1 byte add 1 byte padding
            if element[3] < 256:
                string.extend(bytearray([0]))
                string.extend(bytearray([element[3]]))
            # If it's less than 2 bytes add 1 bytes padding
            elif 256 <= element[3] <= 65536:
                string.extend(self.__bitstring_to_bytes("{0:016b}".format(element[3])))
            # If it's larger than 2 bytes, no padding required

            # Function can only handle ts of under 65536 currently
            else:
                # string.extend((bytearray(['{0:024b}'.format(element[3])])))
                print("Function cannot handle timestamps > 65536 currently")
                return "Error"

        # Set AER flag to True so that other functions cannot use their operations
        self.__AER = True
        self.data = string

    def save_data(self, PATH):
        """Function to save processed data
        Arguments
        ---------
        PATH:       string
                        Path to save location for data. NOTE: Currently you must specify the filename and type in this PATH string
        """
        with open(PATH, "wb") as pickle_out:
            pickle.dump(self.data, pickle_out)
            pickle_out.close()

    def save_data_np(self, PATH):
        np.save(PATH, self.data, allow_pickle=True)

    def offset_values(self, offset, downsample=None, reduce=False):
        """Function to offset input data to avoid negative spike times
        Arguments
        ---------
        offset:     int
                        offset to add to each event in ms
        downsample: bool
                        Enabling downsampling here rounds each event to the nearest 10ms, deleting duplicates
        reduce:     bool
                        Set to true if data is already offset and should be clipped back rather than forwards. Default = False
        """
        data_y, data_x = self.data.shape

        # Loop through array
        for y in range(data_y):
            for x in range(data_x):
                temp_list = []
                # If list isn't empty
                if self.data[y, x] != []:
                    # If need to clip data back then remove all before offset and reduce all values by offset
                    if reduce:
                        temp_list = [spike for spike in self.data[y, x] if spike >= offset]
                        temp_lise = [(spike - offset) for spike in temp_list]
                    # Else simply add the offset to each spike
                    else:
                        temp_list = [(spike + offset) for spike in self.data[y, x]]
                
                # This rounds to the nearest 10ms for downsampling purposes
                if downsample is not None:
                    temp_list = set([round(spike, -1) for spike in temp_list])
                
                self.data[y, x] = temp_list

    def create_events(self, ON_OFF=1):
        """Function to convert data to a lava SLAYER compatible Event object
        Arguments
        ---------
        ON_OFF:     int (default = 1)
                        Int either 0 or 1 to indicate the channel of the data. For some reason SLAYER reads in my data with 0 events (wanted 1 events)
        """
        if not self.__AER:
            # Find all non-zero elements in the 2D array
            non_empty_mask = np.array(
                [[arr.size > 0 for arr in row] for row in self.data]
            )
            indices_of_non_empty = np.argwhere(non_empty_mask)

            # Preallocate arrays based on total number of spikes
            total_spikes = sum(arr.size for arr in self.data[non_empty_mask])
            ts_array = np.empty(
                total_spikes, dtype=int
            )  # Assuming data has uniform dtype
            x_array = np.empty(total_spikes, dtype=int)
            y_array = np.empty(total_spikes, dtype=int)

            # Populate arrays
            index = 0
            for y, x in indices_of_non_empty:
                size = self.data[y, x].size
                ts_array[index : index + size] = self.data[y, x]
                x_array[index : index + size] = x
                y_array[index : index + size] = y
                index += size

            # Sort all values by spike time using NumPy's argsort
            sorted_indices = np.argsort(ts_array)

            # Use these indices to reorder all three arrays
            ts_array = ts_array[sorted_indices]
            x_array = x_array[sorted_indices]
            y_array = y_array[sorted_indices]

            # # Create lists of each of the x,y,spike and c values
            channel_array = np.zeros(len(x_array))

            # Combine arrays into Event object
            # CHWT format
            td_event = slayer.io.Event(
                x_event=x_array,
                y_event=y_array,
                c_event=channel_array,
                t_event=ts_array,
                payload=None,
            )

            self.data = td_event

            return td_event

        else:
            raise ValueError(
                "Data is in AER format and cannot perform create_tensor operation"
            )

    def threshold_pooling(self, kernel_size, stride, threshold):
        """
        Method to apply threshold spiking pooling to the data. Returns the data as a resized (x,y) array.

        Arguments
        ----------
        kernel_size (tuple): Shape of pooling kernel
        stride (int): Stride of pooling operation
        threshold (int): Threshold number of spikes required for chip to retain spike
        """
        # Find the location of chip starting locations
        chips_y, chips_x, out_vector = self.__find_chips(stride)
        chips = list(zip(chips_y, chips_x))

        # Apply padding if needed
        pad_y = (kernel_size[0] - self.data.shape[0] % stride) % kernel_size[0]
        pad_x = (kernel_size[1] - self.data.shape[1] % stride) % kernel_size[1]
        # NOTE: This padding needs to be done with 0s or there must be a catch
        self.data = np.pad(
            self.data, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=None
        )

        # Replace None with empty lists
        for y in range(self.data.shape[0]):
            for x in range(self.data.shape[1]):
                if self.data[y, x] is None:
                    self.data[y, x] = []

        # Create output vector
        data_out = np.zeros(out_vector, dtype=object)

        # Loop through each chip and apply pooling operation
        for chip in range(len(chips)):
            data_out[chip] = self.__find_square(chips[chip], kernel_size, threshold)

        # Reshape vector to 2d
        pooled_dim_y = self.__calculate_pooling_dim(
            self.data.shape[0], kernel_size[0], stride, 1
        )
        pooled_dim_x = self.__calculate_pooling_dim(
            self.data.shape[1], kernel_size[1], stride, 1
        )
        self.data = np.reshape(data_out, (pooled_dim_y, pooled_dim_x))

        return self.data

    def __find_chips(self, stride):
        """
        Private method used in pooling operation to find starting locations for each chip

        Arguments
        ----------

        stide (int): Stride of the pooling kernel over the arrary
        """
        cam_shape = self.data.shape

        y = np.arange(0, cam_shape[0], stride)
        x = np.arange(0, cam_shape[1], stride)

        y_len = len(y)
        x_len = len(x)

        y = np.repeat(y, x_len)
        x = np.tile(x, y_len)

        return y, x, y_len * x_len

    def __find_square(self, start, kernel_size, threshold):
        """
        Private method to create a square around a starting position. With the starting position at the top left of the square
        Based on the number of spikes that occur within this square in the input data, return a 1 or 0

        Arguments
        ----------

        start (tuple): Starting position from whence to draw the square
        kernel_size (tuple): Shape of pooling kernel
        threshold (int): Threshold number of spikes required for chip to retain spike
        """
        # Find all elements in square
        start_y, start_x = start
        kernel_y = np.arange(start_y, start_y + kernel_size[0])
        kernel_x = np.arange(start_x, start_x + kernel_size[1])
        kernel_y = np.repeat(kernel_y, kernel_size[0])
        kernel_x = np.tile(kernel_x, kernel_size[1])

        # Grab all data from kernel
        kernel = self.data[kernel_y, kernel_x]

        # Combine all lists
        combined_list = []
        for lists in kernel:
            combined_list = combined_list + lists

        # Find any repeated values
        vals, counts = np.unique(combined_list, return_counts=True)
        # Append spike times that occur more often than the threshold
        timings = vals[counts >= threshold]

        return timings

    def __calculate_pooling_dim(self, input_dim, kernel_dim, stride, order):
        """
        Private method to calculate the size of a dimension after pooling

        Arguments
        ----------
        input_dim (int): Size of input dimension
        kernel_dim (int): Size of the pooling kernel along the same dimension
        stride (int): Stride of the pooling operation
        order (int): Number of pooling operations being performed
        """
        return int(((input_dim - kernel_dim) / stride) + (1 * order))

    # TODO: Fix this
    def new_threshold_pooling(self, kernel_size, stride, threshold, order=1):
        # Apply padding to the data
        pad_y = (kernel_size[0] - self.data.shape[0] % stride) % kernel_size[0]
        pad_x = (kernel_size[1] - self.data.shape[1] % stride) % kernel_size[1]
        # NOTE: This padding needs to be done with 0s or there must be a catch
        self.data = np.pad(
            self.data, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=None
        )

        # Replace None with empty lists
        for y in range(self.data.shape[0]):
            for x in range(self.data.shape[1]):
                if self.data[y, x] is None:
                    self.data[y, x] = []

        # Calculate output dimensions
        pooled_dim_y = self.__calculate_pooling_dim(
            self.data.shape[0], kernel_size[0], stride, order
        )
        pooled_dim_x = self.__calculate_pooling_dim(
            self.data.shape[1], kernel_size[1], stride, order
        )
        # pooled_dim_y, pooled_dim_x = self.data.shape
        data_out = np.zeros((pooled_dim_y, pooled_dim_x), dtype=object)

        # Initialize data_out with empty lists
        for y in range(pooled_dim_y):
            for x in range(pooled_dim_x):
                data_out[y, x] = []

        # Apply threshold pooling using vectorized operations
        for y in range(0, self.data.shape[0] - kernel_size[0] + 1, stride):
            for x in range(0, self.data.shape[1] - kernel_size[1] + 1, stride):
                square = self.data[y : y + kernel_size[0], x : x + kernel_size[1]]

                # print(square)

                unique_timestamps, counts = np.unique(square, return_counts=True)
                # print(unique_timestamps)
                # Implement thresholding logic here
                # Check if any timestamp exceeds the threshold
                for timestamp, count in zip(unique_timestamps, counts):
                    if count > threshold:
                        data_out[y // stride, x // stride].append(timestamp)

        self.data = data_out

    def add_salt_and_pepper_noise(
        self, sample_length=1000, salt_prob=0.01, pepper_prob=0.01
    ):
        """
        Private method to add salt and pepper noise to data. This "salt" adds in spikes, whereas "pepper" removes spikes.

        Arguments
        ----------
        sample_length (int): Length of the data sample in ms
        salt_prob (float): Probability of adding a spike into a specfic coordinate
        pepper_prob (float): Probability of removing a spike from a specfic coordinate
        """
        noisy_data = np.copy(self.data)
        shape = [self.data.shape[0], self.data.shape[1], sample_length]

        # Total points should be relative to the total number of spikes
        unique_points = np.concatenate(
            [ar for ar in noisy_data.flatten() if len(ar) > 0]
        )
        # print(unique_points)
        total_points = len(unique_points)

        # Salt noise
        num_salt = int(np.ceil(salt_prob * total_points))
        for _ in range(num_salt):
            x_coord = np.random.randint(0, shape[0])
            y_coord = np.random.randint(0, shape[1])
            ts_coord = np.random.randint(0, shape[2])

            # Add in new values and sort in ts order
            noisy_data[x_coord, y_coord] = np.append(
                noisy_data[x_coord, y_coord], ts_coord
            )
            noisy_data[x_coord, y_coord] = np.sort(noisy_data[x_coord, y_coord])

        # Pepper noise
        num_pepper = int(np.ceil(pepper_prob * total_points))
        for _ in range(num_pepper):
            x_coord = np.random.randint(0, shape[0])
            y_coord = np.random.randint(0, shape[1])

            noisy_data[x_coord, y_coord] = np.array([])

        self.data = noisy_data

    def create_lava_array(self, sample_length):
        """
        Method to create a tensor of events compatible with lava input processes

        Arguments
        ----------
        sample_length (int): Length of the sample. Used to determine 3rd dimension of output tensor
        """
        y_size, x_size = self.data.shape
        events = self.create_events()

        event_tensor = events.fill_tensor(
            torch.zeros(1, y_size, x_size, sample_length, requires_grad=False)
        )

        return event_tensor.reshape(-1, sample_length)

    def find_start(self, threshold):
        # Flatten and unpack data
        flat_data = self.data.flatten()
        unpacked_data = [elt for l in flat_data for elt in l]
        sorted_data = sorted(unpacked_data)

        # Find where the threshold spikes occur
        spike_count = 0
        for spike in sorted_data:
            spike_count += 1
            if spike_count >= threshold:
                return spike

# Testing of the class
if __name__ == "__main__":
    path_to_data = "/home/farscope2/Documents/PhD/First_Year_Project/SpikingNetsTexture/datasets/TacTip_NM/ntac_2.5_11texture_100trial_slide_test_06101340/Artificial Dataset 9Texture No. 10.pickle"
    OUTPUT_PATH = "/home/farscope2/Documents/PhD/Tactile_Lava/utils/test.bin"

    # Import data to the DataProcessor class
    proc = DataProcessor.load_data(path=path_to_data)

    import time

    start = time.time()
    proc.threshold_pooling(kernel_size=(2, 2), stride=2, threshold=2)
    end = time.time()
    print(f"Time taken = {end-start}")
