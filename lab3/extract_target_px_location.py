import cv2
import numpy as np
import os

def extract_target_px_location(image, plot=False):
    """Takes opencv BGR format image and extract the pixel location of circle centers in that image

    Args:
        image (opencv image): BGR format using cv2.imread()
        plot (bool, optional): Switch for plotting. Defaults to False.

    Returns:
        circle_px_coords: a list of circle pixel coordinates in tuple (x, y), if no circle in an image then return -1
        image: undistorted image with labelled circles 
    """
    
    greyed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(greyed_image, (3, 3), sigmaX=0)
    rows = blurred_image.shape[0]
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, rows / 16, param1=100, param2=32, minRadius=10, maxRadius=26)
    circle_px_coords = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
                center = (i[0], i[1])
                circle_px_coords.append(center)
                cv2.circle(image, center, 1, (0, 100, 100), 3)
                radius = i[2]
                cv2.circle(image, center, radius, (255, 0, 255), 3)
    
    else:
        circle_px_coords = -1
    return circle_px_coords, image
    
if __name__ == "__main__":
    output_folder = "lab3/output_folder"
    image_files = [f for f in os.listdir(output_folder) if f.endswith(".jpg")]
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) 
    for filename in image_files:
        image_path = os.path.join(output_folder, filename)
        image = cv2.imread(image_path)
        circle_px_coordinates, image = extract_target_px_location(image, True)
        print(f"{filename}: {circle_px_coordinates}")
        cv2.imshow("Image", image)
        cv2.waitKey(15)  # Display each image for 20 ms
    cv2.destroyAllWindows()