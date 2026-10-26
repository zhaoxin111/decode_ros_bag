"""Minimal model input loader. Run: python load_example.py outputs/20261026_144347"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def iter_samples(directory):
    root = Path(directory)
    calib = json.loads((root / 'calibration.json').read_text())
    k = np.asarray(calib['K_rectified'], dtype=np.float64)
    mask = cv2.imread(str(root / calib['rectified_valid_mask']), cv2.IMREAD_GRAYSCALE) > 0
    with (root / 'frames.jsonl').open() as index:
        for line in index:
            frame = json.loads(line)
            files = frame['files']
            bgr = cv2.imread(str(root / files['image_rectified']))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with np.load(root / files['cloud_aligned'], allow_pickle=False) as cloud:
                xyz = cloud['xyz'].copy()
            yield dict(id=frame['id'], rgb=rgb, xyz_camera=xyz,
                       sparse_depth=np.load(root / files['depth'], allow_pickle=False),
                       K=k.copy(), T_world_camera=np.asarray(frame['T_world_camera']),
                       image_valid_mask=mask.copy(), timestamp_ns=frame['image_timestamp_ns'])


if __name__ == '__main__':
    sample = next(iter_samples(sys.argv[1]))
    for key, value in sample.items():
        print(key, f'{value.shape} {value.dtype}' if isinstance(value, np.ndarray) else value)
