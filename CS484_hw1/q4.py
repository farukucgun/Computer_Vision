import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
Discuss your results, are they always perfect? Why so, why not?
"""


def otsu_threshold(image):
    pass


def main():
    img1 = cv2.imread('../images/Q4_a.png', 0)
    img2 = cv2.imread('../images/Q4_b.png', 0)

    # Apply Otsu's thresholding
    binary_image1 = otsu_threshold(img1)
    binary_image2 = otsu_threshold(img2)


if __name__ == "__main__":
    main()