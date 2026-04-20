import os
from is_windows_system import is_windows_system

DIRNOW = os.path.dirname(os.path.abspath(__file__))
GO_ICP_DIR = os.path.join(DIRNOW, "Go-ICP")
GO_ICP_EXE = os.path.join(DIRNOW, "GoIcp." + (
    "exe" if is_windows_system() else "out"
))

GO_ICP_CPP_LIST = [
    os.path.join(GO_ICP_DIR, src_file)
    for src_file in [
        "ConfigMap.cpp",
        "jly_3ddt.cpp",
        "jly_goicp.cpp",
        "jly_main.cpp",
        "matrix.cpp",
        "StringTokenizer.cpp"
    ]
]
