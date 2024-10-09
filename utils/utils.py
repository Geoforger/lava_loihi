import numpy as np
import re
import math
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

def linear_accel(start, dt, rate):
    return start + (rate * dt)

def distance_from_accel(v, t, a):
    return v*t + 1/2*a*t^2

def calculate_velocity_with_distance(initial_velocity, acceleration, distance):
    """
    Calculate velocity with constant acceleration and distance.

    Parameters:
    - initial_velocity (float): Initial velocity.
    - acceleration (float): Acceleration.
    - distance (float): Distance.

    Returns:
    float: Final velocity.
    """
    final_velocity_squared = initial_velocity**2 + 2 * acceleration * distance
    final_velocity = math.sqrt(final_velocity_squared)
    return final_velocity


def time_from_accel(start_velocity, acceleration, distance):
    """
    Function to calculate total time from starting velocity, constant acceleration and distance traveled using quadratic equation

    Parameters
    ----------

    v (int): Starting velocity in mm/s
    a (int): Constant acceleration in mm/s/s
    d (int): Distance traveled in mm

    Returns
    -------

    root1 or root2 (float): Length of the sample in s
    """
    # Calculate the coefficients for the quadratic equation
    a = 0.5 * acceleration
    b = start_velocity
    c = -distance

    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    # Check if the discriminant is non-negative (real solutions)
    if discriminant >= 0:
        # Calculate the two possible solutions for time
        root1 = (-b + np.sqrt(discriminant)) / (2*a)
        root2 = (-b - np.sqrt(discriminant)) / (2*a)

        # Choose the positive root (time cannot be negative)
        if root1 >= 0:
            return root1
        elif root2 >= 0:
            return root2
        else:
            # If both roots are negative, no valid solution
            return None
    else:
        # If the discriminant is negative, no real solutions
        return None
    

def nums_from_string(string):
    """
    Function to return a list of float values from a string
    
    Arguments
    ----------
    string (str): String to extract floats from

    Returns
    ---------
    i (list): List of int values from the string
    """
    l = re.findall(r'\d+', string)
    i = [int(a) for a in l]
    
    return i

def calculate_pooling_dim(input_dim, kernel_dim, stride, order):
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

def dataset_split(PATH, train_ratio=0.6, valid_ratio=None):
    """ Function to split a given directory of data into a training and test split after seperating data for validation

    Args:
        PATH (str): Path to the data directory 
        train_ratio (float, optional): Ratio of training to testing data. Defaults to 0.8.
    """
    filenames = glob.glob(f"{PATH}/*on.pickle.npy")
    
    if os.path.exists(f"{PATH}/train/") and os.path.exists(f"{PATH}/test/"):
        if (input(f"Train & Test directories exist on dataset path {PATH}. Overwrite? This WILL overwrite both directories (y,N)") != "y"):
            print("Not overwriting current directories")
            return
            
    os.makedirs(f"{PATH}/train/", exist_ok=False)
    os.makedirs(f"{PATH}/test/", exist_ok=False)
        
    # Create the train/test/split
    train, test = train_test_split(filenames, train_size=train_ratio, test_size=1-train_ratio)
    
    if valid_ratio is not None:
        assert(type(valid_ratio) is float)
        os.makedirs(f"{PATH}/valid/", exist_ok=False)
        
        # Calculate valid ration of the remaining training data
        test_valid = 1.0 - train_ratio
        val_ratio = valid_ratio / test_valid
        test, valid = train_test_split(test, train_size=val_ratio, test_size=val_ratio)
        
        for f in valid:
            f_s = f.split("/")[-1]
            shutil.copy(f, f"{PATH}/valid/{f_s}")

    # Copy files into folders
    for f in train:
        f_s = f.split("/")[-1]
        shutil.copy(f, f"{PATH}/train/{f_s}")
    for f in test:
        f_s = f.split("/")[-1]
        shutil.copy(f, f"{PATH}/test/{f_s}")

    print("Split files into train test folders")