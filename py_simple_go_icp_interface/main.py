import numpy as np

import time
import tempfile
import uuid
import os
from typing import Optional

_PointCloud = np.ndarray
_RotationMatrix = np.ndarray
_TranslationVector = np.ndarray

try:
    from .utils import GO_ICP_EXE
    from .compile import compile_go_icp
    from .check import check_point_cloud
    from .make_input import save_point_cloud_txt, save_config_txt
    from .run_cmd import run_executable
    from .evaluation import apply_transform, max_abs
except:
    from utils import GO_ICP_EXE
    from compile import compile_go_icp
    from check import check_point_cloud
    from make_input import save_point_cloud_txt, save_config_txt
    from run_cmd import run_executable
    from evaluation import apply_transform, max_abs

def shuffle_rows(arr: np.ndarray, rng:np.random.Generator) -> np.ndarray:
    idx = rng.permutation(arr.shape[0])
    return arr[idx]

def go_icp_match(
        reference_pts: _PointCloud,
        moving_pts   : _PointCloud,
        random_seed  : Optional[int],
        random_downsample : Optional[int] = None,
        rotMinX               : float =  -3.1416,
        rotMinY               : float =  -3.1416,
        rotMinZ               : float =  -3.1416,
        rotWidth              : float =   6.2832,
        MSEThresh             : Optional[float] = None,
        transMinX             : Optional[float] = None,
        transMinY             : Optional[float] = None,
        transMinZ             : Optional[float] = None,
        transWidth            : Optional[float] = None,
        trimFraction          : float =   0.0000,
        distTransExpandFactor : float =   2.0000,
        distTransSize : int = 300,
        force_recompile:bool=False
    ) -> tuple[_RotationMatrix, _TranslationVector]:

    # sovle total time cost
    begin_time = time.time()

    # zero means do not downsample
    if random_downsample is None:
        random_downsample = 0
    random_downsample = max(random_downsample, 0)

    # check point cloud
    check_point_cloud(reference_pts)
    check_point_cloud(moving_pts)

    # random shuffle
    if random_downsample > 0:
        rng       = np.random.default_rng(random_seed)
        reference_pts = shuffle_rows(reference_pts, rng)
        moving_pts  = shuffle_rows( moving_pts, rng)

    # compile .exe/.out file
    compile_go_icp(force_recompile)

    # create temporary directory (auto clean)
    with tempfile.TemporaryDirectory(
        prefix="py_go_icp_" + uuid.uuid4().hex[:16] + "_") as temp_dir:

        # get abs folder
        if not os.path.isabs(temp_dir):
            temp_dir = os.path.abspath(temp_dir)

        # relevant files
        model_txt  = os.path.join(temp_dir, "model.txt")
        data_txt   = os.path.join(temp_dir, "data.txt")
        config_txt = os.path.join(temp_dir, "config.txt")
        output_txt = os.path.join(temp_dir, "output.txt")

        # save point cloud
        save_point_cloud_txt(reference_pts, model_txt)
        save_point_cloud_txt(moving_pts, data_txt)

        # get transform redius
        tran_radius = max(
            max_abs(reference_pts), max_abs(moving_pts))
        if transMinX is None:
            transMinX = -tran_radius
        if transMinY is None:
            transMinY = -tran_radius
        if transMinZ is None:
            transMinZ = -tran_radius
        if transWidth is None:
            transWidth = 2 * tran_radius
        if MSEThresh is None:
            assert transWidth is not None
            MSEThresh = transWidth * 0.001

        # save config
        save_config_txt(
            MSEThresh,
            rotMinX,rotMinY,rotMinZ,rotWidth,
            transMinX,transMinY,transMinZ,transWidth,
            trimFraction,distTransExpandFactor,distTransSize,
            config_txt)
        
        # run program
        ret = run_executable(
            GO_ICP_EXE,
            [model_txt, data_txt, str(random_downsample), config_txt, output_txt]
        )

        # raise runtime error
        if ret != 0:
            raise RuntimeError("Go-Icp runtime error")

        # acquire R and T
        r_mat = []
        t_vec = []
        with open(output_txt, "r") as fpin:
            for line in fpin:
                if len(r_mat) < 3:
                    r_mat.append(list(map(float, line.split())))
                else:
                    t_vec.append(list(map(float, line.split())))

        r_mat = np.array(r_mat).astype(np.float32)
        t_vec = np.array(t_vec).astype(np.float32).T

        # output total time cost
        total_time_cost = time.time() - begin_time
        print(f"Total time cost: {total_time_cost:.3f}s")
        return (r_mat, t_vec)

if __name__ == "__main__":
    import test_data # reply on open3d
    src, tgt = test_data.load_test_data()

    # Get rotation and transform vector
    r_mat, t_vec = go_icp_match(tgt, src, random_seed=127, force_recompile=True)
    print(r_mat, t_vec)
    
    # visualization
    rot_src    = apply_transform(src, r_mat, t_vec)
    robust_dis = test_data.robust_dist_q3_max(rot_src, tgt)
    test_data.visualize_two_point_clouds(rot_src, tgt)
