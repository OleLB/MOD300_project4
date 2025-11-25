from Task3 import plt2rgbarr
import numpy as np

def encode_rgb_data(rgb_array):
    """Function to encode rgb data into brightness levels."""
    brightness_encoding = np.zeros(rgb_array.shape[:2], dtype=int)

    # Calculate brightness as the sum of RGB values
    brightness = np.sum(rgb_array, axis=2)

    # Define brightness levels
    brightness_encoding[brightness < 250] = 0  # Dark
    brightness_encoding[(brightness >= 250) & (brightness < 400)] = 1  # Medium
    brightness_encoding[brightness >= 400] = 2  # Bright

    return brightness_encoding



if __name__ == "__main__":
    from Task2 import gen_milkyway_sector

    # Generate sector view centered at "Omega Centauri"
    fig = gen_milkyway_sector("Omega Centauri", 5000, save=False)

    # Convert the figure to an RGB array
    rgb_array = plt2rgbarr(fig)

    # Create categories based on pixel brightness
    brightness_encoding = encode_rgb_data(rgb_array)

    print(f"Categories shape: {brightness_encoding.shape}")
    print(f"Unique categories: {np.unique(brightness_encoding)}")
    # print count of each category
    unique, counts = np.unique(brightness_encoding, return_counts=True)
    print(f"Category counts: {dict(zip(unique, counts))}")

    # Plot the categories
    import matplotlib.pyplot as plt
    plt.imshow(brightness_encoding) # reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    # plt.colorbar(ticks=[0, 1, 2], label='Category')
    plt.show()