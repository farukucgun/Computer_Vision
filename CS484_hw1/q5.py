import numpy as np
import cv2

from q1 import dilation, erosion
from q2_3 import create_histogram, equalize_histogram, apply_histogram_equalization
from q4 import otsu_threshold


"""
Objective

The goal is to identify and count the distinct objects in each provided grayscale image through a series
of image processing steps. The process involves thresholding to separate objects from the background,
morphological operations to refine the object shapes, and connected components labeling to label distinct
objects.

Procedure
1. Thresholding: Find a threshold that produces a binary image, effectively separating objects from
the background. Specify the method used for determining the threshold (e.g., trial and error or Otsu’s
method) and justify your choice.
2. Morphological Operations: Use the implemented morphological operators (dilation, erosion, and
their combinations) to separate objects connected together or to fill holes within the objects. Consider
the use of inverse operations on the binary image, especially if the objects are lighter than the
background.
3. Labelling: With a refined binary image, employ connected components analysis to produce a labelled
image, assigning a unique label (an integer ID) to each distinct region.

Restrictions: You MUST use your own implementations for dilation, erosion, and thresholding. However,
you CAN use external libraries (such as OpenCV) for arithmetic and logical operations, connected components
labelling, and image I/O functions. You are NOT allowed to use any advanced functions beyond the
specified operations.

Provide an analytical justification for the sequence of operations performed. Explain what each step achieves
in the context of object labelling and counting.
"""


def main():
    # Read the image
    image = cv2.imread('../images/Q5.png', 0)

    # Apply Otsu's thresholding
    binary_image = otsu_threshold(image)

    # Define structuring element (arbitrary shape)
    structuring_element = np.array([[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
    
    # Apply erosion
    eroded_image = erosion(binary_image, structuring_element)
    cv2.imwrite('output_images/A.png', eroded_image)

    # Apply dilation
    dilated_image = dilation(eroded_image, structuring_element)

    # Apply connected components labeling
    num_labels, labeled_image = cv2.connectedComponents(dilated_image)
    labeled_image = labeled_image + 1

    # display the number of objects
    print("Number of objects: ", num_labels - 1)

    # save the labeled image
    cv2.imwrite('output_images/Q5_labeled.png', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()