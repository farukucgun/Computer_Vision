import numpy as np
import cv2

from q1 import erosion, dilation
from q2_3 import create_histogram, equalize_histogram, apply_histogram_equalization
from q4 import otsu_threshold


def main():
    # Read the image
    image = cv2.imread('../images/Q5.png', 0)

    # Apply Otsu's thresholding
    binary_image = otsu_threshold(image)
    cv2.imwrite('output_images/Q5_thresholded.png', binary_image)

    # Define structuring element (arbitrary shape)
    structuring_element = np.array([[255, 0, 255],
                                    [0, 0, 0],
                                    [255, 0, 255]])
    
    # Apply dilation
    dilated_image = dilation(binary_image, structuring_element)
    cv2.imwrite('output_images/Q5_dilated.png', dilated_image)
    
    # Apply erosion
    eroded_image = erosion(dilated_image, structuring_element)
    cv2.imwrite('output_images/Q5_eroded.png', eroded_image)

    # Apply connected components labeling
    num_labels, labeled_image = cv2.connectedComponents(eroded_image)
    labeled_image = labeled_image + 1

    # display the number of objects
    print("Number of objects: ", num_labels - 1)

    # save the labeled image
    cv2.imwrite('output_images/Q5_labeled.png', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()