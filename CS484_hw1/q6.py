import numpy as np
import cv2


def two_d_convolution(image, filter):
    # Get dimensions of the image and the filter
    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape

    # Calculate padding size for the image
    pad_height = filter_height // 2
    pad_width = filter_width // 2

    # Create an empty output image
    output_image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for i in range(pad_height, image_height - pad_height):
        for j in range(pad_width, image_width - pad_width):
            # Fit the filter in the image at this pixel
            area_of_interest = image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]

            # Perform the convolution operation
            output_image[i, j] = np.sum(area_of_interest * filter)

    return output_image


def sobel_operator(image):
    # Define the Sobel operator
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Apply the Sobel operator
    sobel_x_image = two_d_convolution(image, sobel_x)
    sobel_y_image = two_d_convolution(image, sobel_y)

    # Find the magnitude of the gradient
    gradient_magnitude = np.sqrt(sobel_x_image ** 2 + sobel_y_image ** 2)

    return sobel_x_image, sobel_y_image, gradient_magnitude


def prewitt_operator(image):
    # Define the Prewitt operator
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
    
    prewitt_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])

    # Apply the Prewitt operator
    prewitt_x_image = two_d_convolution(image, prewitt_x)
    prewitt_y_image = two_d_convolution(image, prewitt_y)

    # Find the magnitude of the gradient
    gradient_magnitude = np.sqrt(prewitt_x_image ** 2 + prewitt_y_image ** 2)

    return prewitt_x_image, prewitt_y_image, gradient_magnitude


def main():
    # import the image
    image = cv2.imread('../images/Q6.png', 0)

    # scale down the image to range [0, 1]
    image = image / 255.0

    # Apply the Sobel operator
    sobel_x_image, sobel_y_image, sobel_edges = sobel_operator(image)

    # Save the results
    cv2.imwrite('output_images/Q6_sobel_edges_x.png', sobel_x_image)
    cv2.imwrite('output_images/Q6_sobel_edges_y.png', sobel_y_image)
    cv2.imwrite('output_images/Q6_sobel_edges.png', sobel_edges * 255)
    
    # Apply the Prewitt operator
    prewitt_x_image, prewitt_y_image, prewitt_edges = prewitt_operator(image)

    # Save the results
    cv2.imwrite('output_images/Q6_prewitt_edges_x.png', prewitt_x_image)
    cv2.imwrite('output_images/Q6_prewitt_edges_y.png', prewitt_y_image)
    cv2.imwrite('output_images/Q6_prewitt_edges.png', prewitt_edges * 255)


if __name__ == "__main__":
    main()