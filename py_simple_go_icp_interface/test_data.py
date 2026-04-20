import numpy as np

try:
    from scipy.spatial import KDTree
except:
    KDTree = None

o3d = None
try:
    import open3d as o3d
except:
    pass

def robust_dist_q3_max(pc_a: np.ndarray, pc_b: np.ndarray) -> float:
    """
    Calculate the robust distance between two point clouds:
    1. Calculate nearest neighbor distances from A→B and B→A
    2. Combine all distances
    3. Calculate Q1 (25%) and Q3 (75%)
    4. Return the larger value between Q1 and Q3 (usually Q3)

    Args:
        pc_a: shape (N, 3)
        pc_b: shape (M, 3)

    Returns:
        max(Q1, Q3)
    """
    if KDTree is None:
        raise ModuleNotFoundError("Module scipy not found.")
    
    # Build KDTree
    tree_b = KDTree(pc_b)
    tree_a = KDTree(pc_a)

    # Nearest neighbor distances
    dist_a2b, _ = tree_b.query(pc_a, k=1)
    dist_b2a, _ = tree_a.query(pc_b, k=1)

    # Combine all distances
    all_dists = np.concatenate([dist_a2b, dist_b2a])

    # Quartiles
    q1 = np.percentile(all_dists, 25)
    q3 = np.percentile(all_dists, 75)

    return max(float(q1), float(q3))

def load_bunny(down_sample:float=0.1, noise:float=0.0):
    if o3d is None:
        raise ModuleNotFoundError("Module open3d not found.")
    dataset = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)
    pcd = mesh.sample_points_uniformly(number_of_points=int(35000*down_sample))
    pts = np.asarray(pcd.points)
    pts -= pts.mean(axis=0)
    if noise > 0:
        pts += np.random.normal(0, noise, pts.shape)
    return pts

def make_reg_pair(
        src:np.ndarray, 
        seed:int=42, 
        trans:tuple[float, float, float]=(0.2,0.1,0.05), 
        rot_deg:tuple[float, float, float]=(15.0,10.0,5.0), 
        noise:float=0.003
    ):

    rng = np.random.default_rng(seed)
    def rot_mat(deg, axis) -> np.ndarray:
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        if axis == 'z':
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        assert False, "Invalid axis, must be x, y, or z"
    
    R = rot_mat(rot_deg[2], 'z') @ rot_mat(rot_deg[1], 'y') @ rot_mat(rot_deg[0], 'x')
    T = np.array(trans)
    tgt = src @ R.T + T
    if noise > 0:
        tgt += rng.normal(0, noise, tgt.shape)
    return src, tgt, R, T

def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    bunny = load_bunny(down_sample=0.2, noise=0.001)
    src, tgt, R_true, T_true = make_reg_pair(bunny, seed=42, rot_deg=(-90, 35, -45.5))
    return src, tgt

def visualize_two_point_clouds(
    source: np.ndarray,
    target: np.ndarray,
    color_source: list = [1, 0, 0],    # Red
    color_target: list = [0, 1, 0],    # Green
    point_size: int = 2,
) -> None:
    """
    Visualize two point clouds simultaneously (Open3D window)
    
    Args:
        source: Source point cloud, shape (N, 3)
        target: Target point cloud, shape (M, 3)
        color_source: Color of source point cloud [r, g, b], range 0~1
        color_target: Color of target point cloud [r, g, b], range 0~1
        point_size: Size of the rendered points
    """
    if o3d is None:
        raise ModuleNotFoundError("Module open3d not found.")
    
    # Source point cloud
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(source)
    pcd_src.paint_uniform_color(color_source)

    # Target point cloud
    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(target)
    pcd_tgt.paint_uniform_color(color_target)

    # Visualization
    assert hasattr(o3d, "visualization")
    vis = o3d.visualization.Visualizer() # type:ignore
    vis.create_window()
    vis.add_geometry(pcd_src)
    vis.add_geometry(pcd_tgt)

    # Set point size
    opt = vis.get_render_option()
    opt.point_size = point_size

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    src, tgt = load_test_data()
    print(f"src: {src.shape}, tgt: {tgt.shape}")
    visualize_two_point_clouds(src, tgt)
