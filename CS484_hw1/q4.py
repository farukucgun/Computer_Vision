import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
Discuss your results, are they always perfect? Why so, why not?
"""

# find the threshold t that minimizes the weighted sum of within-class variance for the two groups
# resulting from separating the gray levels at t
def otsu_threshold(image):
    # Compute histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Normalize histogram
    hist = hist.ravel() / hist.sum()

    # Compute the cumulative sum of the normalized histogram
    P = np.cumsum(hist)

    # Compute the cumulative sum of the normalized histogram multiplied by the intensity
    P_times_i = np.cumsum(hist * np.arange(256))

    # Compute the cumulative sum of the normalized histogram multiplied by the intensity squared
    P_times_i_squared = np.cumsum(hist * np.arange(256) ** 2)

    # Compute the cumulative mean
    mean = P_times_i[-1]

    # Compute the cumulative variance
    variance = P_times_i_squared[-1] - mean ** 2

    # Initialize variables
    threshold = 0
    max_variance = 0

    # Iterate through each intensity level
    for i in range(256):
        # Compute the probability of class 1
        P1 = P[i]

        # Compute the probability of class 2
        P2 = 1 - P1
        
        # Compute the mean of class 1
        mean1 = P_times_i[i] / P1

        # Compute the mean of class 2
        mean2 = (mean - P_times_i[i]) / P2

        # Compute the variance of class 1
        variance1 = P_times_i_squared[i] - mean1 ** 2

        # Compute the variance of class 2
        variance2 = (P_times_i_squared[-1] - P_times_i_squared[i]) - mean2 ** 2

        # Compute the within-class variance
        within_class_variance = P1 * variance1 + P2 * variance2

        # Compute the between-class variance
        between_class_variance = variance - within_class_variance

        # Update the threshold if the between-class variance is greater than the maximum variance
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = i

    # Apply the threshold to the image
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255

    # Display the binary image
    # cv2.imshow('Binary Image', binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return binary_image
    

def main():
    img1 = cv2.imread('../images/Q4_a.png', 0)
    img2 = cv2.imread('../images/Q4_b.png', 0)

    # Apply Otsu's thresholding
    binary_image1 = otsu_threshold(img1)
    binary_image2 = otsu_threshold(img2)

    # Save the binary images
    cv2.imwrite('output_images/Q4_a_otsu.png', binary_image1)
    cv2.imwrite('output_images/Q4_b_otsu.png', binary_image2)

if __name__ == "__main__":
    main()