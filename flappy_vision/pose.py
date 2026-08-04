"""Pose tracking and flap detection - the heart of the project.

The signal chain is:

    landmarks -> arm elevation H (in shoulder-widths)   scale-invariant
              -> one-euro filter                        jitter out, lag low
              -> velocity dH/dt
              -> Schmitt trigger + refractory           one flap, one event

Measuring H in shoulder-widths is what makes the thresholds hold whether you are
1m or 3m from the camera. Triggering on downward *velocity* rather than on a
static "wrist above shoulder" line is what makes it feel like flapping rather
than like holding a pose.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config as cfg
from .capture import open_camera

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
ARM_POINTS = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
ARM_BONES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16))

MIN_VISIBILITY = 0.5


class _LowPass:
    def __init__(self) -> None:
        self.y: float | None = None

    def __call__(self, x: float, a: float) -> float:
        self.y = x if self.y is None else a * x + (1 - a) * self.y
        return self.y


class OneEuroFilter:
    """Adaptive low-pass: heavy smoothing when still, light when moving fast.

    A plain EMA forces a single tradeoff - smooth but laggy, or responsive but
    jittery. A flap needs both: steady when the arm is held, instant on the
    downstroke. Also exposes the filtered derivative, which is what the trigger
    actually keys on.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._x_prev: float | None = None
        self._t_prev: float | None = None
        self.dx_hat = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.__init__(self.min_cutoff, self.beta, self.d_cutoff)

    def __call__(self, x: float, t: float) -> tuple[float, float]:
        """Return (smoothed value, smoothed derivative)."""
        if self._t_prev is None:
            self._t_prev, self._x_prev = t, x
            self.dx_hat = 0.0
            return self._x(x, 1.0), 0.0

        dt = max(t - self._t_prev, 1e-4)
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        self._x_prev = x
        self.dx_hat = self._dx(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(self.dx_hat)
        return self._x(x, self._alpha(cutoff, dt)), self.dx_hat


class AdaptiveBand:
    """Decaying envelope of recent arm elevation.

    Fixed thresholds assume the player keeps their arms in the same place all
    game. They don't: arms sag as you tire, and people drift between "big wing
    flaps" and "hold them out and flap the forearms". Once you leave the band,
    a fixed trigger goes silent and the game feels broken.

    Tracking the recent min/max instead makes the trigger care about the *shape*
    of the motion rather than its absolute height, so a small tired flap at hip
    level works the same as a big one overhead.
    """

    def __init__(self, tau: float = 2.0, min_span: float = 0.30) -> None:
        self.tau = tau  # seconds of motion the envelope remembers
        self.min_span = min_span  # below this, treat it as "not flapping"
        self.lo: float | None = None
        self.hi: float | None = None

    @property
    def span(self) -> float:
        if self.lo is None:
            return 0.0
        return self.hi - self.lo

    @property
    def active(self) -> bool:
        """False when the arms aren't moving enough to be a deliberate flap."""
        return self.span >= self.min_span

    def reset(self) -> None:
        self.lo = self.hi = None

    def update(self, h: float, dt: float) -> None:
        if self.lo is None:
            self.lo = self.hi = h
            return
        # Instant attack, exponential release toward the current value. Decay
        # has to be *relative* (a time constant) rather than a fixed rate: a
        # fixed px/sec release outruns a small-amplitude flap, collapsing the
        # envelope between strokes so gentle flapping stops registering.
        k = math.exp(-dt / self.tau)
        self.hi = max(h, h + (self.hi - h) * k)
        self.lo = min(h, h + (self.lo - h) * k)

    def thresholds(self, up_frac: float, down_frac: float) -> tuple[float, float]:
        return (self.lo + up_frac * self.span, self.lo + down_frac * self.span)


class FlapState(Enum):
    DOWN = auto()  # arms low, waiting for a raise
    ARMED = auto()  # arms up, waiting for the downstroke
    RECOVER = auto()  # just fired, waiting for arms to come back down


class FlapDetector:
    """Schmitt trigger on arm elevation, firing on the downstroke.

    Separate up/down thresholds (hysteresis) plus a refractory window are what
    stop a single flap registering as several - landmark jitter sitting right on
    a single threshold is exactly how naive implementations double-fire.
    """

    def __init__(self, conf: cfg.DetectorConfig | None = None) -> None:
        self.conf = conf or cfg.DetectorConfig()
        self.filter = OneEuroFilter(self.conf.min_cutoff, self.conf.beta)
        self.band = AdaptiveBand(min_span=self.conf.min_span)
        self.state = FlapState.DOWN
        self.h = 0.0
        self.v = 0.0
        self._fired_at = -1e9
        self._t_prev: float | None = None
        # Live thresholds, exposed so the HUD can draw where the band actually is.
        self.h_up = self.conf.h_up
        self.h_down = self.conf.h_down

    def reset(self) -> None:
        self.filter.reset()
        self.band.reset()
        self.state = FlapState.DOWN
        self._fired_at = -1e9
        self._t_prev = None

    def update(self, h_raw: float, t: float) -> bool:
        """Feed one elevation sample. Returns True on the frame a flap fires."""
        self.h, self.v = self.filter(h_raw, t)
        c = self.conf

        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 1e-4)
        self._t_prev = t

        if c.adaptive:
            self.band.update(self.h, dt)
            if not self.band.active:
                # Not enough motion to be a flap; hold state and don't fire.
                self.h_up, self.h_down = self.band.thresholds(c.up_frac, c.down_frac)
                return False
            self.h_up, self.h_down = self.band.thresholds(c.up_frac, c.down_frac)
            v_gate = max(c.v_floor, c.v_frac * self.band.span)
        else:
            self.h_up, self.h_down = c.h_up, c.h_down
            v_gate = c.v_thresh

        # The refractory window gates *firing*; coming back down is what gates
        # *re-arming*. Requiring both to re-arm deadlocks the detector at fast
        # flap rates, because the brief arms-low window falls entirely inside
        # the refractory period and the re-arm condition never coincides.
        cooled = (t - self._fired_at) >= c.refractory_s

        if c.mode == "raise":
            # Simpler fallback: fire on the rising edge of a static threshold.
            if self.state is FlapState.DOWN and self.h > self.h_up and cooled:
                self.state = FlapState.RECOVER
                self._fired_at = t
                return True
            if self.state is FlapState.RECOVER and self.h < self.h_down:
                self.state = FlapState.DOWN
            return False

        if self.state is FlapState.DOWN:
            if self.h > self.h_up:
                self.state = FlapState.ARMED
        elif self.state is FlapState.ARMED:
            if self.v < -v_gate and cooled:
                self.state = FlapState.RECOVER
                self._fired_at = t
                return True
            if self.h < self.h_down:  # lowered arms without a real downstroke
                self.state = FlapState.DOWN
        elif self.state is FlapState.RECOVER:
            if self.h < self.h_down:
                self.state = FlapState.DOWN
        return False


def arm_elevation(landmarks) -> float | None:
    """Height of the higher wrist above the shoulder line, in shoulder-widths.

    Returns None when the landmarks needed are not confidently visible, so the
    caller can hold the last state instead of acting on garbage.
    """
    needed = (L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST)
    if any(landmarks[i].visibility < MIN_VISIBILITY for i in needed):
        return None

    ls, rs = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
    width = math.hypot(ls.x - rs.x, ls.y - rs.y)
    if width < 1e-3:  # degenerate: person edge-on or mis-detected
        return None

    shoulder_y = (ls.y + rs.y) / 2
    # y grows downward, so subtracting gives positive-is-higher.
    return max(
        (shoulder_y - landmarks[L_WRIST].y) / width,
        (shoulder_y - landmarks[R_WRIST].y) / width,
    )


@dataclass
class Snapshot:
    """Latest vision state, published for the render loop to read."""

    frame: np.ndarray
    points: dict[int, tuple[float, float]] = field(default_factory=dict)
    have_pose: bool = False
    h: float = 0.0
    v: float = 0.0
    state: FlapState = FlapState.DOWN
    t_capture: float = 0.0
    infer_ms: float = 0.0
    vision_fps: float = 0.0
    # Live band, which moves with your recent motion in adaptive mode.
    h_up: float = 0.0
    h_down: float = 0.0
    span: float = 0.0
    active: bool = False


@dataclass
class FlapEvent:
    t_capture: float  # when the frame was grabbed - for true end-to-end latency
    t_fired: float


class VisionThread:
    """Capture + inference on one background thread.

    Capture blocks ~33ms at 30fps while inference costs ~8ms, so a single thread
    never falls behind and every frame it hands over is fresh. The render loop
    reads the latest snapshot and never blocks on either.
    """

    def __init__(
        self,
        camera: int = 0,
        conf: cfg.DetectorConfig | None = None,
        record: bool = False,
    ) -> None:
        self.detector = FlapDetector(conf)
        self._camera_index = camera
        self._snapshot: Snapshot | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.events: deque[FlapEvent] = deque(maxlen=32)
        self.error: Exception | None = None
        self.trace: list[dict] | None = [] if record else None

    def start(self) -> None:
        # Open the camera on the calling thread so permission errors surface
        # before the UI comes up, rather than dying silently in a worker.
        self._cap = open_camera(self._camera_index)
        self._landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(cfg.POSE_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:  # surfaced to the main thread
            self.error = exc

    def _loop(self) -> None:
        fps_ema = 30.0
        t_last = time.perf_counter()

        while not self._stop.is_set():
            ok, frame = self._cap.read()
            t_capture = time.perf_counter()
            if not ok or frame is None:
                continue

            # Mirror before inference so drawn landmarks line up with what the
            # player sees. Left/right labels swap, which is fine - we take the
            # higher of the two arms anyway.
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            result = self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                # Real monotonic timestamps: faking 30fps feeds the tracker wrong
                # motion priors whenever the camera drifts off that rate.
                int(t_capture * 1000),
            )
            infer_ms = (time.perf_counter() - t0) * 1000

            points: dict[int, tuple[float, float]] = {}
            have_pose = bool(result.pose_landmarks)
            h_raw = None
            if have_pose:
                lms = result.pose_landmarks[0]
                points = {
                    i: (lms[i].x, lms[i].y)
                    for i in ARM_POINTS
                    if lms[i].visibility >= MIN_VISIBILITY
                }
                h_raw = arm_elevation(lms)

            if h_raw is not None:
                if self.detector.update(h_raw, t_capture):
                    self.events.append(FlapEvent(t_capture, time.perf_counter()))
                if self.trace is not None:
                    self.trace.append({"t": t_capture, "h": h_raw})

            dt = t_capture - t_last
            t_last = t_capture
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 / dt

            snap = Snapshot(
                frame=frame,
                points=points,
                have_pose=have_pose and h_raw is not None,
                h=self.detector.h,
                v=self.detector.v,
                state=self.detector.state,
                t_capture=t_capture,
                infer_ms=infer_ms,
                vision_fps=fps_ema,
                h_up=self.detector.h_up,
                h_down=self.detector.h_down,
                span=self.detector.band.span,
                active=self.detector.band.active,
            )
            with self._lock:
                self._snapshot = snap  # overwrite, never queue - queuing is lag

    def latest(self) -> Snapshot | None:
        with self._lock:
            return self._snapshot

    def drain_flaps(self) -> list[FlapEvent]:
        """Pop all flaps since the last call.

        Deliberately a queue rather than a latest-value flag: a flap landing
        between two rendered frames must not vanish.
        """
        out = []
        while self.events:
            out.append(self.events.popleft())
        return out

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
