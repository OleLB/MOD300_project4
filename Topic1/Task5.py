"""Use K-means and k-nn clustering to categorize RGB data."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def knn_clustering(rgb_array, brightness_encoding):
    """
    Perform k-nearest neighbors clustering on the given data.

    Parameters
    ----------
    rgb_array : np.ndarray, array of rgb data
    brightness_encoding : np.ndarray, array of brightness categories (labeled data)
    
    Sources:
    https://www.w3schools.com/python/python_ml_knn.asp
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    knn = KNeighborsClassifier(n_neighbors=3)

    # Reshape data for scikit-learn
    pixel_count = rgb_array.shape[0] * rgb_array.shape[1]
    rgb_array = rgb_array.reshape((pixel_count, 3))

    # brightness_labels_flat = categories_array.flatten()
    brightness_encoding = brightness_encoding.flatten()

    knn.fit(rgb_array, brightness_encoding) 
    clustered_data = knn.predict(rgb_array)
    return clustered_data


def kmeans_clustering(rgb_array, n_clusters=3):
    """
    Perform k-means clustering on the given RGB data.

    Parameters
    ----------
    rgb_array: The raw image data (not encoded)
    n_clusters : The number of clusters it should group the data into

    Returns:
        np.ndarray: An array containing the cluster label (0, 1, or 2) for each pixel.

    Sources:
    https://www.w3schools.com/python/python_ml_k-means.asp
    scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 
    """
    # Reshaping the RGB data, must be 2D for KMeans
    n_pixels = rgb_array.shape[0] * rgb_array.shape[1] # calc number of pixels
    rgb_data_flat = rgb_array.reshape(n_pixels, 3) # 3 because of RGB

    # Set up KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    kmeans.fit(rgb_data_flat)

    clustered_data = kmeans.predict(rgb_data_flat)

    return clustered_data


if __name__ == "__main__":
    from Topic1.Task2 import gen_milkyway_sector
    from Topic1.Task3 import plt2rgbarr

    # Generate sector view centered at "Omega Centauri"
    fig = gen_milkyway_sector("Omega Centauri", 2500, save=False)

    # Convert the figure to an RGB array
    test_rgb_array = plt2rgbarr(fig)

    # Create categories based on pixel brightness
    from Topic1.Task4 import encode_rgb_data_brightness
    test_brightness_encoding = encode_rgb_data_brightness(test_rgb_array)

    # Perform K-NN clustering
    test_clustered_data = kmeans_clustering(test_rgb_array, test_brightness_encoding)

    # Reshape clustered data back to image shape
    clustered_image = test_clustered_data.reshape((
        test_rgb_array.shape[0], test_rgb_array.shape[1]))

    print(f"Clustered image shape: {clustered_image.shape}")
    print(f"Unique clusters: {np.unique(clustered_image)}")

    # Plot the clustered image
    import matplotlib.pyplot as plt
    plt.imshow(clustered_image)
    plt.show()
