# **AER1217 Lab 3: Georeferencing Using UAV Payload Data**

## **Step-by-Step Lab Workflow**

### **1. Fix Image Distortion**
- The given **camera intrinsic parameters** include distortion coefficients.
- Use **OpenCV** to undistort the images before processing:
  
```python
import cv2
import numpy as np

# Camera intrinsic matrix K
K = np.array([[698.86, 0.0, 306.91],
              [0.0, 699.13, 150.34],
              [0.0, 0.0, 1.0]])

# Distortion coefficients
dist_coeffs = np.array([0.191887, -0.563680, -0.003676, -0.002037, 0.000000])

# Load image
image = cv2.imread("image.jpg")

# Undistort the image
undistorted_img = cv2.undistort(image, K, dist_coeffs)

# Save or display the undistorted image
cv2.imshow("Undistorted", undistorted_img)
cv2.waitKey(0)
```

---

### **2. Extract Target Pixel Location**
- Use **OpenCV** to detect the circular target in the image.
- Possible methods:
  - **Hough Circle Transform** (`cv2.HoughCircles()`)
  - **Contour Detection** (`cv2.findContours()`)

```python
gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect circles
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=50, param2=30, minRadius=10, maxRadius=50)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])  # Pixel coordinates (u, v)
        radius = i[2]
        cv2.circle(undistorted_img, center, radius, (0, 255, 0), 2)

cv2.imshow("Detected Target", undistorted_img)
cv2.waitKey(0)
```

---

### **3. Transform Pixel Coordinates to Vicon Frame**
#### **(a) Image Coordinates → 3D Camera Frame**
- Use the **camera intrinsic matrix (K)** to get normalized coordinates:
  $X_c = \frac{(u - c_x)}{f_x}, \quad Y_c = \frac{(v - c_y)}{f_y}, \quad Z_c = 1$
  
  - $( u, v )$ = detected pixel location
  - $( (c_x, c_y) = (306.91, 150.34) )$ (principal point from K)
  - $( (f_x, f_y) = (698.86, 699.13) )$ (focal lengths)
$
#### **(b) Camera Frame → Body Frame**
- Use the given **extrinsic transformation matrix $( T_{CB} )$**:
  
  $T_{CB} =
  \begin{bmatrix}
  0 & -1 & 0 & 0 \\
  -1 & 0 & 0 & 0 \\
  0 & 0 & -1 & 0 \\
  0 & 0 & 0 & 1
  \end{bmatrix}$
  
  - Multiply the **3D camera coordinates** by $( T_{CB}^{-1} )$.

#### **(c) Body Frame → Vicon Frame**
- Get the **drone pose (position and quaternion) from the CSV** at the matching timestamp.
- Convert **quaternion to rotation matrix** to get **$T_{BV}$**.
- Apply **homogeneous transformation** to move from **Body → Vicon**.

---

### **4. Track Targets Across Multiple Images**
- Store all detected targets in Vicon coordinates.
- Apply **DBSCAN clustering** to group detections that belong to the **same target**.

```python
from sklearn.cluster import DBSCAN
import numpy as np

# List of target detections in Vicon frame
detections = np.array([...])  # (X_v, Y_v, Z_v) from multiple images

# Apply DBSCAN clustering
clustering = DBSCAN(eps=0.1, min_samples=2).fit(detections)

# Cluster labels
labels = clustering.labels_
```

---

### **5. Compute Final Target Positions**
- Average the detected **Vicon positions** of each cluster to get a **final estimate** for each target.

```python
final_positions = {}
for cluster in set(labels):
    if cluster != -1:
        cluster_points = detections[labels == cluster]
        final_positions[cluster] = np.mean(cluster_points, axis=0)

print("Final target positions:", final_positions)
```

