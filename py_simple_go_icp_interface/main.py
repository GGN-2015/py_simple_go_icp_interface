import numpy as np
from live_chrono import LiveChrono

import tempfile
import uuid
import os
from typing import Optional
from threading import Thread
from queue import Queue

_PointCloud = np.ndarray
_RotationMatrix = np.ndarray
_TranslationVector = np.ndarray

try:
    from .utils import GO_ICP_EXE
    from .compile import compile_go_icp
    from .check import check_point_cloud
    from .make_input import save_point_cloud_txt, save_config_txt
    from .run_cmd import run_executable
    from .evaluation import apply_transform
except:
    from utils import GO_ICP_EXE
    from compile import compile_go_icp
    from check import check_point_cloud
    from make_input import save_point_cloud_txt, save_config_txt
    from run_cmd import run_executable
    from evaluation import apply_transform


def shuffle_rows(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(arr.shape[0])
    return arr[idx]


def go_icp_match_raw(
        reference_pts: _PointCloud,
        moving_pts: _PointCloud,
        random_seed: Optional[int],
        random_downsample: Optional[int] = None,
        rotMinX: float = -3.141593,
        rotMinY: float = -3.141593,
        rotMinZ: float = -3.141593,
        rotWidth: float = 6.283185,
        trimFraction: float = 0.000000,
        distTransExpandFactor: float = 2.000000,
        distTransSize: int = 300,
        force_recompile: bool = False,
        print_output=False
) -> tuple[_RotationMatrix, _TranslationVector]:

    if random_downsample is None:
        random_downsample = 0
    random_downsample = max(random_downsample, 0)

    check_point_cloud(reference_pts)
    check_point_cloud(moving_pts)

    if random_downsample > 0:
        rng = np.random.default_rng(random_seed)
        reference_pts = shuffle_rows(reference_pts, rng)
        moving_pts = shuffle_rows(moving_pts, rng)

    all_pts = np.vstack([reference_pts, moving_pts])
    min_xyz = all_pts.min(axis=0)
    max_xyz = all_pts.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0

    max_extent = np.max(max_xyz - min_xyz)
    scale = (max_extent / 2.0) * 10.0
    scale = max(scale, 1e-8)

    ref_norm = (reference_pts - center) / scale
    mov_norm = (moving_pts - center) / scale

    compile_go_icp(force_recompile, print_output)

    with tempfile.TemporaryDirectory(prefix="py_go_icp_" + uuid.uuid4().hex[:16] + "_") as temp_dir:
        if not os.path.isabs(temp_dir):
            temp_dir = os.path.abspath(temp_dir)

        model_txt = os.path.join(temp_dir, "model.txt")
        data_txt = os.path.join(temp_dir, "data.txt")
        config_txt = os.path.join(temp_dir, "config.txt")
        output_txt = os.path.join(temp_dir, "output.txt")

        save_point_cloud_txt(ref_norm, model_txt)
        save_point_cloud_txt(mov_norm, data_txt)

        # Can not change these parameters
        MSEThresh: float = 0.000010
        transMinX: float = -0.100000
        transMinY: float = -0.100000
        transMinZ: float = -0.100000
        transWidth: float = 0.200000

        save_config_txt(
            MSEThresh,
            rotMinX, rotMinY, rotMinZ, rotWidth,
            transMinX, transMinY, transMinZ, transWidth,
            trimFraction, distTransExpandFactor, distTransSize,
            config_txt
        )

        ret = run_executable(
            GO_ICP_EXE,
            [model_txt, data_txt, str(random_downsample), config_txt, output_txt],
            print_output=print_output
        )

        if ret != 0:
            raise RuntimeError("Go-ICP runtime error")

        r_mat = []
        t_vec = []
        with open(output_txt, "r") as fpin:
            for line in fpin:
                line = line.strip()
                if not line:
                    continue
                if len(r_mat) < 3:
                    r_mat.append(list(map(float, line.split())))
                else:
                    t_vec.append(list(map(float, line.split())))

        R_norm = np.array(r_mat, dtype=np.float32)
        t_norm = np.array(t_vec, dtype=np.float32).reshape(3, 1)

        t_ori = scale * t_norm + (np.eye(3) - R_norm) @ center.reshape(3, 1)
        R_ori = R_norm
        t_ori = t_ori.ravel()

    return R_ori, t_ori


# ===================== Multi-threaded Robust Matching Function =====================
def go_icp_match(
        reference_pts: _PointCloud,
        moving_pts: _PointCloud,
        random_seed: Optional[int] = None,
        random_downsample: Optional[int] = None,
        rotMinX: float = -3.141593,
        rotMinY: float = -3.141593,
        rotMinZ: float = -3.141593,
        rotWidth: float = 6.283185,
        trimFraction: float = 0.000000,
        distTransExpandFactor: float = 2.000000,
        distTransSize: int = 300,
        force_recompile: bool = False,
        print_output=True,
        n_threads=3
) -> tuple[_RotationMatrix, _TranslationVector]:
    """
    Multi-threaded robust GO-ICP matching:
    Run go_icp_match_raw in N parallel threads and select the result with the highest robust_q3 score.
    """

    # At least one thread
    n_threads = max(n_threads, 1)

    compile_go_icp(force_recompile, print_output)

    # Result queue for inter-thread communication
    result_queue = Queue()
    threads = []

    # Generate distinct random seeds for each thread
    if random_seed is None:
        seeds = [None] * n_threads
    else:
        seeds = [random_seed + i for i in range(n_threads)]

    # Worker function for each thread
    def worker(seed):
        try:
            R, t = go_icp_match_raw(
                reference_pts=reference_pts,
                moving_pts=moving_pts,
                random_seed=seed,
                random_downsample=random_downsample,
                rotMinX=rotMinX,
                rotMinY=rotMinY,
                rotMinZ=rotMinZ,
                rotWidth=rotWidth,
                trimFraction=trimFraction,
                distTransExpandFactor=distTransExpandFactor,
                distTransSize=distTransSize,
                force_recompile=False,
                print_output=False
            )
            # Compute robust q3 score
            transformed = apply_transform(moving_pts, R, t)
            errors = np.linalg.norm(transformed - reference_pts, axis=1)
            q3 = np.percentile(errors, 75)
            robust_q3 = -q3  # lower q3 = better, so we use negative for sorting
            result_queue.put((robust_q3, R, t))

        except Exception as e:
            print(f"[Thread Error] seed={seed}: {str(e)}")

    # Timer
    chrono = LiveChrono(display_format="[Go-ICP threads] Elapsed: %H:%M:%S")
    chrono.start()
    
    # Start all threads
    for s in seeds:
        t = Thread(target=worker, args=(s,))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()
    chrono.stop()

    # Collect valid results
    candidates = []
    while not result_queue.empty():
        score, R, t = result_queue.get()
        candidates.append((score, R, t))

    if len(candidates) == 0:
        raise RuntimeError("All GO-ICP threads failed to execute")

    # Select result with the highest robust_q3 (smallest q3 error)
    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_R, best_t = candidates[0]

    if print_output:
        print(f"[Robust Match] Selected best result (q3={-best_score:.4f})")
        print(f"Valid results: {len(candidates)}/{n_threads}")

    return best_R, best_t


if __name__ == "__main__":
    import test_data
    src, tgt = test_data.load_test_data()

    r_mat, t_vec = go_icp_match(
        tgt, src, 
        random_seed=127, force_recompile=True)

    print("Rotation:\n", r_mat)
    print("Translation:\n", t_vec)

    rot_src = apply_transform(src, r_mat, t_vec)
    test_data.visualize_two_point_clouds(rot_src, tgt)
