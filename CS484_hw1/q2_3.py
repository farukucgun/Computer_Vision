import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def create_histogram(image):
    # Create an empty histogram
    histogram = np.zeros(256, dtype=np.uint32)

    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Increment the corresponding bin in the histogram
            histogram[image[i, j]] += 1

    return histogram


def visualize_histogram(histogram, title):
    # Create an array of bins
    bins = np.arange(256)

    # Plot the histogram
    plt.bar(bins, histogram, color='gray')
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Save and display the histogram
    plt.savefig("output_images/" + title)
    plt.show()


def calculate_cum_sum(hist, value):
    cum_sum = np.zeros_like(hist)
    cum_sum[0] = hist[0]
    
    for i in range(1, 256):
        cum_sum[i] = cum_sum[i - 1] + hist[i] * math.pow(i, value)

    return cum_sum


def equalize_histogram(histogram):
    # Create an empty lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)

    # Calculate the cumulative distribution function
    cdf = calculate_cum_sum(histogram, 0)

    # Normalize the CDF
    cdf_normalized = cdf / cdf[-1]

    # Create the lookup table
    for i in range(256):
        lookup_table[i] = np.round(255 * cdf_normalized[i])

    return lookup_table


def apply_histogram_equalization(image, lookup_table):
    # Apply the lookup table on the image manually 
    equalized_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = lookup_table[image[i, j]]

    return equalized_image


def main():
    # Read the images
    img1 = cv2.imread("../images/Q2_a.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("../images/Q2_b.png", cv2.IMREAD_GRAYSCALE)
    
    # Create the histograms
    hist1 = create_histogram(img1)
    hist2 = create_histogram(img2)

    # Visualize the histograms
    visualize_histogram(hist1, "Q2_a_histogram.jpg")
    visualize_histogram(hist2, "Q2_b_histogram.png")

    # Equalize the histograms
    lookup_table1 = equalize_histogram(hist1)
    lookup_table2 = equalize_histogram(hist2)

    # Apply the histogram equalization
    equalized_img1 = apply_histogram_equalization(img1, lookup_table1)
    equalized_img2 = apply_histogram_equalization(img2, lookup_table2)

    # Save the equalized images
    cv2.imwrite("output_images/Q3_a_histogram_equalized.jpg", equalized_img1)
    cv2.imwrite("output_images/Q3_b_histogram_equalized.png", equalized_img2)

    # Create the histograms of the equalized images
    hist1_eq = create_histogram(equalized_img1)
    hist2_eq = create_histogram(equalized_img2)

    # Visualize the histograms of the equalized images
    visualize_histogram(hist1_eq, "Q3_a_histogram_equalized.jpg")
    visualize_histogram(hist2_eq, "Q3_b_histogram_equalized.png")


if __name__ == "__main__":
    main()