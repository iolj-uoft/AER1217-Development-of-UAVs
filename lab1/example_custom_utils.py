"""Example utility module.

Please use a file like this one to add extra functions.

"""
import numpy as np

def exampleFunction():
    """Example of user-defined function.

    """
    x = -1
    return x

def fx(x0, y0, r, timesteps):
    theta = np.linspace(0 - np.pi / 2, 2 * np.pi - np.pi / 2, timesteps)
    return r * np.cos(theta) + x0

def fy(x0, y0, r, timesteps):
    theta = np.linspace(0 - np.pi / 2, 2 * np.pi - np.pi / 2, timesteps)
    return r * np.sin(theta) + y0
