from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def registered_pair_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_masks = np.zeros((2, 4, 6), dtype=bool)
    ref_masks[0, 0:2, 0:2] = True
    ref_masks[1, 1:3, 3:5] = True

    mov_masks_raw = np.zeros_like(ref_masks)
    mov_masks_raw[0, 0:2, 1:3] = True
    mov_masks_raw[1, 1:3, 4:6] = True

    mov_masks_registered = np.zeros_like(ref_masks)
    mov_masks_registered[0, 0:2, 0:2] = True
    mov_masks_registered[1, 1:3, 3:5] = True

    return ref_masks, mov_masks_raw, mov_masks_registered


@pytest.fixture
def write_raw_npy_session():
    def _write_raw_npy_session(
        subject_dir: Path,
        session_name: str,
        roi_masks: np.ndarray,
        *,
        offset: float,
        plane_name: str = "plane0",
    ) -> Path:
        plane_dir = subject_dir / session_name / "data_npy" / plane_name
        plane_dir.mkdir(parents=True, exist_ok=True)
        np.save(plane_dir / "rois.npy", roi_masks)
        np.save(
            plane_dir / "F.npy",
            np.array([[offset, offset + 1], [offset + 2, offset + 3]], dtype=float),
        )
        np.save(plane_dir / "fov.npy", np.full(roi_masks.shape[1:], offset, dtype=float))
        return plane_dir

    return _write_raw_npy_session
