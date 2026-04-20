# py_simple_go_icp_interface
A simple python encapsulation of Go-ICP.

> [!NOTE]
>
> This project only provides a simple Python interface encapsulation for the C++ project.
>
> For the core algorithm, please refer to: [https://github.com/yangjiaolong/Go-ICP](https://github.com/yangjiaolong/Go-ICP)
>
> Cython implementation, see: [https://github.com/aalavandhaann/go-icp_cython](https://github.com/aalavandhaann/go-icp_cython)

## Installation

> [!IMPORTANT]
>
> Make sure you have available `g++` (or mingw for windows) on your `PATH`, `Go-ICP` program will be compiled at runtime.

```bash
pip install py_simple_go_icp_interface
```

## Usage

```python
from py_simple_go_icp_interface import go_icp_match, apply_transform
import numpy as np

reference_points = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
])
moving_points = np.array([
    [ 0.78947212, -1.62329061, 1.90248941],
    [ 0.20479484, -0.90372354, 1.52782194],
    [ 0.67225195, -1.24123766, 2.81916545],
    [ 1.59222483, -1.04341237, 1.7634595 ]
])

# get rotation and transform matrix
r_mat, t_vec = go_icp_match(reference_points, moving_points, random_seed=42)

# move point set
moved_points = apply_transform(moving_points, r_mat, t_vec)

# output moved point set
print(moved_points)
```
