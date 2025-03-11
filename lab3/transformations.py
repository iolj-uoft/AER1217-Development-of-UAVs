import numpy as np
from scipy.spatial.transform import Rotation as R

def pixel2bodyFrame(pixel_coordinates):
    """Transform pixel coordinates to the Vicon (global) frame

    Args:
        pixel_coordinates (u, v): a tuple for the pixel coordinates for target center
        pose_csv (np.array): an numpy array contains the 3D location (x, y) and quaternion (qw, qx, qy, qz) coordinates
    """
    # Initialize parameters
    u, v = pixel_coordinates
    c_x = 306.91
    c_y = 150.34
    f_x = 698.86
    f_y = 699.13
    
    T_CB = np.array([[0, -1, 0, 0],
                     [-1, 0, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])
    T_BC = np.linalg.inv(T_CB)

    # Calculate the camera frame coordinates
    x_c = (u - c_x) / f_x
    y_c = (v - c_y) / f_y
    z_c = 1
    w_c = 1
    P_C = np.array([x_c, y_c, z_c, w_c])
    
    # Transform coordinates from camera frame -> body frame
    P_B = T_BC @ P_C
    P_B /= P_B[-1] # Normalize with the homogeneous term
    
    return P_B

def body2ViconFrame(P_B, pose):
    """Transform P_B to P_V

    Args:
        P_B (size 4 np.array): [x_b, y_b, z_b, 1]
        pose (size 7 np.array): [p_x, p_y, p_z, q_w, q_x, q_y, q_z]

    Returns:
        P_V (size 3 np.array): [x_v, y_v, z_v]
    """
    # Get the drone pose from .csv file and form the rotation matrix from quaternion coordinates
    x, y, z, qw, qx, qy, qz = pose
    R_VB = R.from_quat([qx, qy, qz, qw]).as_matrix()
    P_V = R_VB @ P_B[:3] + np.array([x, y, z])
    
    return P_V
    
if __name__ == "__main__":
    print(pixel2bodyFrame((422, 276)))
    print(body2ViconFrame((-0.17973767, -0.16468248, -1., 1.), (-1.52195,1.36339,0.0943046,0.743646,-1.71519e-06,0.0159058,-0.668384)))
