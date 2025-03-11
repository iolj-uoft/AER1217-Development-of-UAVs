import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from extract_target_px_location import extract_target_px_location
from undistort_image import undistort_image
import transformations as tf

def main(show_video=False):
    pose_csv = pd.read_csv("lab3/lab3_pose.csv")
    pose_csv = pose_csv.iloc[:, 1:].to_numpy()
    
    image_folder = "lab3/output_folder"
    target_coordinates = []
    
    for i in range(pose_csv.shape[0]):
        image_path = os.path.join(image_folder, f"image_{i}.jpg")
        image = cv2.imread(image_path)
        if image is None: 
            print(f"Image {image_path} not found.")
            break
        
        undistorted_image = undistort_image(image)
        target_pixel_coordinates, image = extract_target_px_location(undistorted_image)
        if target_pixel_coordinates != -1:
            for j in range(len(target_pixel_coordinates)):
                P_B = tf.pixel2bodyFrame(target_pixel_coordinates[j])
                P_V = tf.body2ViconFrame(P_B, pose_csv[i])
                if P_V[0] < 2 and P_V[0] > -2 and P_V[1] < 2 and P_V[1] > -2:
                    target_coordinates.append(P_V)
        if show_video:
            cv2.imshow("Image", image)
            cv2.waitKey(10)
    
    detections = np.array(target_coordinates)
    xy_detections = detections[:, :2]  # extract only the x, y coordinates
    
    dbscan = DBSCAN(eps=0.4, min_samples=30, n_jobs=-1).fit(xy_detections)  # apply DBSCAN clustering to label each cluster
    labels = dbscan.labels_  # extract cluster labels
    unique_labels = set(labels)  # get unique cluster IDs

    # Compute mean for each cluster
    cluster_means = {}  
    for cluster in unique_labels:
        if cluster != -1:  # ignore outliers where label == -1)
            cluster_points = xy_detections[labels == cluster]  # get all points in a single cluster
            cluster_means[cluster] = np.mean(cluster_points, axis=0)  # compute mean 

    # Plot results
    plt.figure(figsize=(6, 6))
    for cluster in unique_labels:
        if cluster == -1:
            plt.scatter(xy_detections[labels == cluster, 0], xy_detections[labels == cluster, 1], c="gray", marker="x", label="Outliers") # plot outliers
        else:
            plt.scatter(xy_detections[labels == cluster, 0], xy_detections[labels == cluster, 1], label=f"Target {cluster}", alpha=0.01) # plot clustered points

    # Plot cluster mean positions
    for cluster, mean_pos in cluster_means.items():
        plt.scatter(mean_pos[0], mean_pos[1], c='k', marker='*', s=120, label=f'Cluster {cluster} Mean')
        plt.text(mean_pos[0] + 0.5, mean_pos[1] - 0.3, f'({mean_pos[0]:.4f}, {mean_pos[1]:.4f})', fontsize=9, ha='right')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2) 
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Target Positions with Clusters")
    plt.savefig("lab3/Target Positions.png", dpi=150)
    print("Plot Saved.")
    plt.show()
    
if __name__ == "__main__":
    main(show_video=False)
