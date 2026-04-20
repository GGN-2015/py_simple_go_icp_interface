import numpy as np

def max_abs(arr: np.ndarray) -> float:
    return np.max(np.abs(arr))

def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply rotation and translation transformation (rigid transformation) to an N×3 point cloud.
    
    Transformation formula: P_transformed = P @ R.T + t
    
    Args:
        points: Original point cloud, numpy array with shape (N, 3)
        R: Rotation matrix, numpy array with shape (3, 3)
        t: Translation vector, numpy array with shape (3,) or (1, 3)
    
    Returns:
        Transformed point cloud with shape (N, 3)
    """
    # Input validation (defensive programming to avoid bugs)
    assert points.ndim == 2 and points.shape[1] == 3, "Point cloud must be an N*3 array"
    assert R.shape == (3, 3), "Rotation matrix R must be 3x3"
    assert t.shape == (3,) or t.shape == (1, 3), "Translation vector t must have length 3"
    
    # Core transformation formula (vectorized operation, extremely fast)
    # R.T is used because points are stored as row vectors (N,3), not mathematical column vectors (3,N)
    transformed_points = points @ R.T + t
    
    return transformed_points
