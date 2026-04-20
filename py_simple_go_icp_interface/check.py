import numpy as np

def check_point_cloud(data_arr:np.ndarray) -> None:

    if not isinstance(data_arr, np.ndarray):
        raise TypeError("point cloud should be numpy.ndarray")
    
    if (len(data_arr.shape) != 2) or (data_arr.shape[1] != 3):
        raise TypeError("point cloud should be (N * 3) numpy.ndarray")
    
    if data_arr.shape[0] <= 0:
        raise TypeError("point cloud should not be empty")
