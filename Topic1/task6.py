"""Plotting cluster over original image"""

import matplotlib.pyplot as plt

def overlay_clustered_image(
        original_rgb,
        clustered_data,
        alpha=0.5,
        title="Original Image with K-Means Clustering Overlay"
    ):
    """
    Function to overlay clustered data on the original RGB image.
    Cluster data must already be reshaped for plotting.
    """

    # Plot original image
    plt.imshow(original_rgb)  # array in H x W x 3 format
    plt.title(title)

    # Plot cluster data on top with some transparency
    plt.imshow(
        clustered_data,
        cmap='viridis',
        alpha=alpha
    )

    plt.axis('off') # Hide axis because they are not needed
    plt.show()
