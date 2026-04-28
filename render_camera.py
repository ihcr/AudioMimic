import numpy as np


def compute_axes_limits(poses, min_span=3.0, padding=0.25):
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[-1] != 3:
        raise ValueError("Expected poses with shape [frames, joints, 3]")

    flat = poses.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    span = max(span + 2.0 * float(padding), float(min_span))
    half = span / 2.0

    xlim = (float(center[0] - half), float(center[0] + half))
    ylim = (float(center[1] - half), float(center[1] + half))
    zlim = (float(center[2] - half), float(center[2] + half))
    return xlim, ylim, zlim
