import numpy as np
import cv2


""""
you are expected to apply a sequence of morphological operations to come up with a cleaner version 
of the image given where the noise is removed and the characters are readable.
"""


def threshold_image(image, threshold):
    # Apply thresholding to the image
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return thresholded_image


def erosion(image, kernel):
    # Get dimensions of the image and the kernel
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding size for the image
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create an empty output image
    eroded_image = np.full((img_height, img_width), 255, dtype=np.uint8)

    # Iterate over each pixel in the image
    for i in range(pad_height, img_height - pad_height):
        for j in range(pad_width, img_width - pad_width):
            """
            fit the kernel in the image at this pixel
            area of interest --> area of image overlapping with the area where the kernel has pixel values 0
            if all the pixels in the area of interest have pixel value 255, we set the center pixel to 255
            """ 
            area_of_interest = image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]

            true_so_far = False
            for k in range(kernel_height):
                for l in range(kernel_width):
                    if kernel[k, l] == 0 and area_of_interest[k, l] == 0:
                        true_so_far = True
                    elif kernel [k, l] != 0:
                        continue
                    else:
                        true_so_far = False
                        break

            if true_so_far:
                eroded_image[i, j] = 0

    return eroded_image


def dilation(image, kernel):
    # Get dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding size for the image
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create an empty output image
    dilated_image = np.full((image_height, image_width), 255, dtype=np.uint8)

    # Iterate over each pixel in the image    
    for i in range(pad_height, image_height - pad_height):
        for j in range(pad_width, image_width - pad_width):
            """
            fit the kernel in the image at this pixel
            area of interest --> area of image overlapping with the area where the kernel has pixel values 0
            if we see any pixel value 0 in the area of interest, we set all the pixel values in the area of interest to 0
            """ 
            area_of_interest = image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]

            for k in range(kernel_height):
                for l in range(kernel_width):
                    if kernel[k, l] == 0 and area_of_interest[k, l] == 0:
                        dilated_image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1] = 0
                        break

    return dilated_image


def main ():
    # Define structuring element (arbitrary shape)
    structuring_element = np.array([[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
    
    # Load image (binary format)
    image = cv2.imread('../images/Q1.png', 0)

    # Apply thresholding
    thresholded_img = threshold_image(image, 127)

    cv2.imwrite('output_images/Q1_thresholded.png', thresholded_img)

    # Apply dilation
    dilated_img = dilation(thresholded_img, structuring_element)

    # Apply erosion
    eroded_img = erosion(thresholded_img, structuring_element)

    # Save results
    cv2.imwrite('output_images/Q1_Dilated.png', dilated_img)
    cv2.imwrite('output_images/Q1_Eroded.png', eroded_img)


if __name__ == '__main__':
    main()