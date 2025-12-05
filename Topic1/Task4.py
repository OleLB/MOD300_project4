"""File contains various function to encode rgb data"""

import numpy as np

def encode_rgb_data_brightness(rgb_array):
    """Function to encode rgb data into brightness levels."""
    brightness_encoding = np.zeros(rgb_array.shape[:2], dtype=int)

    # Calculate brightness as the sum of RGB values
    brightness = np.sum(rgb_array, axis=2)

    # Define brightness levels
    brightness_encoding[brightness < 250] = 0  # Dark
    brightness_encoding[(brightness >= 250) & (brightness < 400)] = 1  # Medium
    brightness_encoding[brightness >= 400] = 2  # Bright

    return brightness_encoding


def encode_rgb_data_grey(rgb_array):
    """Grey encoding for the rgb data"""
    luminance_weights = np.array([0.299, 0.587, 0.114]) # Standard luminance weights
    grey = np.sum(rgb_array[:, :, :] * luminance_weights, axis=2)  # From RGB to grey
    x, y = [], []
    for ig, g in enumerate(grey):
        for ij, j in enumerate(g):
            if j > 230:  # Select only bright pixels
                x.append(ig)
                y.append(ij)

    encoding = np.zeros(rgb_array.shape[:2], dtype=int)
    encoding[x, y] = 1  # Mark bright pixels as category 1
    return encoding


if __name__ == "__main__":
    from Topic1.Task3 import plt2rgbarr
    from Topic1.Task2 import gen_milkyway_sector

    # Generate sector view centered at "Omega Centauri"
    fig = gen_milkyway_sector("Omega Centauri", 5000, save=False)

    # Convert the figure to an RGB array
    test_rgb_array = plt2rgbarr(fig)

    # Create categories based on pixel brightness
    test_brightness_encoding = encode_rgb_data_brightness(test_rgb_array)

    print(f"Categories shape: {test_brightness_encoding.shape}")
    print(f"Unique categories: {np.unique(test_brightness_encoding)}")
    # print count of each category
    unique, counts = np.unique(test_brightness_encoding, return_counts=True)
    print(f"Category counts: {dict(zip(unique, counts))}")

    # Plot the categories
    import matplotlib.pyplot as plt
    plt.imshow(test_brightness_encoding)
    plt.show()
