"""Camera discovery and opening.

macOS specifics worth knowing: index 0 is ambiguous when a Continuity Camera
(iPhone) is paired, and the first open triggers a TCC permission prompt against
the *terminal app*, not this script. If permission was denied the capture opens
but yields nothing, so callers should treat "opened but no frames" as a
permission problem rather than a hardware one.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass

import cv2

BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


@contextlib.contextmanager
def _quiet_stderr():
    """Silence OpenCV's noisy probe warnings when scanning camera indices."""
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


@dataclass
class CameraInfo:
    index: int
    width: int
    height: int
    fps: float


def list_cameras(max_index: int = 4) -> list[CameraInfo]:
    found = []
    for i in range(max_index):
        with _quiet_stderr():
            cap = cv2.VideoCapture(i, BACKEND)
            ok = cap.isOpened()
            if ok:
                got, frame = cap.read()
                if got and frame is not None:
                    found.append(
                        CameraInfo(
                            i,
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            float(cap.get(cv2.CAP_PROP_FPS)),
                        )
                    )
            cap.release()
    return found


class CameraError(RuntimeError):
    pass


def open_camera(index: int, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        raise CameraError(
            f"could not open camera {index}.\n"
            f"  - run `flappy --list-cameras` to see what is available\n"
            f"  - on macOS, grant camera access to your terminal in\n"
            f"    System Settings > Privacy & Security > Camera, then restart it"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Keep the driver queue shallow; a deep queue shows up as input lag.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise CameraError(
            f"camera {index} opened but returned no frames - this is almost always\n"
            f"  a macOS permission problem. Grant camera access to your terminal in\n"
            f"  System Settings > Privacy & Security > Camera, then restart it."
        )
    return cap
