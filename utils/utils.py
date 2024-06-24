import numpy as np
import re
import math

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