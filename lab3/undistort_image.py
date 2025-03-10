import cv2
import numpy as np

def undistort_image(image):
    K = np.array([[698.86, 0, 306.91],
                  [0, 699.13, 150.34],
                  [0, 0, 1]])
    dist_coefficient = np.array([0.191887, -0.56368, -0.003676, -0.002037, 0])
    undistorted_image = cv2.undistort(image, K, dist_coefficient)
    
    return undistorted_image

if __name__ == "__main__":
    image = cv2.imread("lab3/output_folder/image_500.jpg")
    image = undistort_image(image)
    cv2.imshow("Undistorted", image)
    cv2.waitKey(0)
    