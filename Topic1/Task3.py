"""Convert the image generated into a rgb np.array (each pixel will be a list of 3 number,
Red, Green, Blue (rbg)) between 0-255"""

import numpy as np
from Topic2.Task2 import gen_milkyway_sector


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
    # Generate sector view centered at "Omega Centauri"
    fig = gen_milkyway_sector("Omega Centauri", 5000, save=False)

    # Convert the figure to an RGB array
    rgb_array = plt2rgbarr(fig)

    print(f"RGB array shape: {rgb_array.shape}")
    print(f"RGB array sample (5 pixels): {rgb_array.reshape(-1, 3)[:5]}")