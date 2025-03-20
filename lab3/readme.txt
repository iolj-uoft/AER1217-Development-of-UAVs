# AER1217 Development of UAVs

This lab focuses on UAV development and georeferencing using UAV payload data. The provided python scripts process images, extract target locations, and transform coordinates into a global reference frame.

## Folder Structure

- **lab3/**: Contains all the scripts, dataset to perform the georeferencing task
  - `undistort_image.py`: Corrects lens distortion in images using camera intrinsic parameters.
  - `extract_target_px_location.py`: Detects circular targets in images and returns their pixel coordinates.
  - `transformations.py`: Contains functions to transform coordinates into different frames.
  - `main.py`: Import the upon 3 python scripts to process images to detect targets, transform their coordinates, and cluster detections using `DBSCAN`, and lastly make the plot for the estimated target location.

## How to Run

1. **Lab 3 Scripts**:
   - Ensure the required dependencies are installed: `numpy`, `opencv-python`, `matplotlib`, `scikit-learn`, `scipy`, and `pandas`.
   - Place the input images in the `lab3/output_folder` directory.
   - Ensure the pose data is in `lab3/lab3_pose.csv`.
   - If there's error finding the required files, go to `main.py` and change the `image_folder_path` and `csv_path` as to your preference.
   - Run `main.py` to process the images and generate the target positions plot.

   ```bash
   python lab3/main.py
   ```

## Dependencies

- Python 3.12
- Required libraries:
  - `numpy`
  - `opencv-python`
  - `matplotlib`
  - `scikit-learn`
  - `pandas`
  - `scipy`

## Outputs

- **Lab 3**:
  - Undistorted images displayed during processing.
  - A plot of clustered target positions saved as `lab3/Target Positions.png`.

## Notes

- Ensure the input images and pose data are correctly formatted and placed in the appropriate directories.
