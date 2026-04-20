import numpy as np

def save_point_cloud_txt(
    point_cloud: np.ndarray,
    filepath: str
):
    n = point_cloud.shape[0]

    with open(filepath, "w", encoding="utf-8") as fpout:
        fpout.write(f"{n}\n")

        for i in range(n):
            x, y, z = point_cloud[i, :]
            fpout.write(f"{x:.15f} {y:.15f} {z:.15f}\n")

def save_config_txt(
    MSEThresh: float,
    rotMinX: float,
    rotMinY: float,
    rotMinZ: float,
    rotWidth: float,
    transMinX: float,
    transMinY: float,
    transMinZ: float,
    transWidth: float,
    trimFraction: float,
    distTransExpandFactor: float,
    distTransSize: int,
    config_txt: str):

    with open(config_txt, "w", encoding="utf-8") as fpout:
        fpout.write(f"MSEThresh={MSEThresh:.15f}\n")
        fpout.write(f"rotMinX={rotMinX:.15f}\n")
        fpout.write(f"rotMinY={rotMinY:.15f}\n")
        fpout.write(f"rotMinZ={rotMinZ:.15f}\n")
        fpout.write(f"rotWidth={rotWidth:.15f}\n")
        fpout.write(f"transMinX={transMinX:.15f}\n")
        fpout.write(f"transMinY={transMinY:.15f}\n")
        fpout.write(f"transMinZ={transMinZ:.15f}\n")
        fpout.write(f"transWidth={transWidth:.15f}\n")
        fpout.write(f"trimFraction={trimFraction:.15f}\n")
        fpout.write(f"distTransExpandFactor={distTransExpandFactor:.15f}\n")
        fpout.write(f"distTransSize={distTransSize:d}\n")
