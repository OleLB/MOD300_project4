"""Convert the image generated into a rgb np.array (each pixel will be a list of 3 number,
Red, Green, Blue (rbg)) between 0-255"""

import numpy as np
from Task2 import gen_milkyway_sector


def plt2rgbarr(fig):
    """
    A function to transform a matplotlib to a 3d rgb np.array 

    Input
    -----
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Output
    ------
    np.array(ndim, ndim, 3): A 3d map of each pixel in a rgb encoding (the three dimensions are x, y, and rgb)
    
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr[:, :, :3]


if __name__ == "__main__":
    fig = gen_milkyway_sector("M8", 4000, save=False)
    rgb_array = plt2rgbarr(fig)
    print(rgb_array.shape)
    print(rgb_array[:5, :5, :])  # Print the first 5x5 pixels