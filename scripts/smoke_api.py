"""Phase 0: verify the MediaPipe 1.0 Tasks API surface before building on it."""

import mediapipe as mp

print("mediapipe", mp.__version__)

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

print("BaseOptions      ", mp_python.BaseOptions)
print("PoseLandmarker   ", vision.PoseLandmarker)
print("PoseLandmarkerOpt", vision.PoseLandmarkerOptions)
print("RunningMode      ", list(vision.RunningMode))
print("mp.Image         ", mp.Image)
print("mp.ImageFormat   ", mp.ImageFormat.SRGB)

# The exact option names the detector will pass.
import inspect

sig = inspect.signature(vision.PoseLandmarkerOptions)
print("PoseLandmarkerOptions params:", list(sig.parameters))
