"""Over imposing cluster plot and original image"""

import numpy as np
import matplotlib.pyplot as plt

def overlay_clustered_image(original_rgb, clustered_data, alpha=0.5):
    """
    Function to overlay clustered data on the original RGB image.
    Cluster data must already be reshaped for plotting.
    """
    plt.figure(figsize=(10, 10))

    # Display the original image
    plt.imshow(original_rgb)  # array in H x W x 3 format
    plt.title("Original Image with K-Means Clustering Overlay")

    # Overimpose the cluster data
    plt.imshow(
        clustered_data, 
        cmap='viridis', 
        alpha=alpha
    )

    cbar = plt.colorbar(shrink=0.7)
    cbar.set_label('K-Means Cluster ID (0, 1, 2)')

    plt.axis('off') # Hide axis because they are not needed
    plt.show()