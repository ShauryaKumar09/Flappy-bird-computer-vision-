"""Threshold calibration.

Fixed thresholds assume an average body at an average distance doing an average
flap. Sampling the player's own arms-down and arms-up elevation and placing the
thresholds inside that range makes the detector fit whoever is standing there.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum, auto

from . import config as cfg

SETTLE_S = 1.5  # time to get into position before sampling starts
COLLECT_S = 2.0  # sampling window per phase
MIN_SAMPLES = 12
MIN_RANGE = 0.4  # below this the two poses are indistinguishable


class Phase(Enum):
    DOWN = auto()
    UP = auto()
    DONE = auto()


PROMPTS = {
    Phase.DOWN: ("ARMS DOWN", "relax at your sides"),
    Phase.UP: ("ARMS UP", "reach as high as you can"),
}


@dataclass
class Calibrator:
    conf: cfg.DetectorConfig
    sens: cfg.Sensitivity = cfg.SENS_NORMAL
    phase: Phase = Phase.DOWN
    elapsed: float = 0.0
    samples: dict[Phase, list[float]] = field(default_factory=lambda: {Phase.DOWN: [], Phase.UP: []})
    failed: str | None = None
    summary: str = ""

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    @property
    def sampling(self) -> bool:
        return self.elapsed >= SETTLE_S

    @property
    def progress(self) -> float:
        """0..1 through the current phase's sampling window."""
        if not self.sampling:
            return 0.0
        return min(1.0, (self.elapsed - SETTLE_S) / COLLECT_S)

    def prompt(self) -> tuple[str, str]:
        if self.done:
            return ("CALIBRATED", "flap to start")
        title, hint = PROMPTS[self.phase]
        if not self.sampling:
            return (title, f"{hint} - {SETTLE_S - self.elapsed:.0f}")
        return (title, "hold still...")

    def update(self, snap, dt: float) -> None:
        if self.done:
            return
        # Only advance while a pose is visible, so stepping out of frame pauses
        # calibration rather than filling it with garbage.
        if snap is None or not snap.have_pose:
            return

        self.elapsed += dt
        if self.sampling:
            self.samples[self.phase].append(snap.h)

        if self.elapsed >= SETTLE_S + COLLECT_S:
            self.elapsed = 0.0
            self.phase = Phase.UP if self.phase is Phase.DOWN else Phase.DONE
            if self.done:
                self._apply()

    def _apply(self) -> None:
        down, up = self.samples[Phase.DOWN], self.samples[Phase.UP]
        if len(down) < MIN_SAMPLES or len(up) < MIN_SAMPLES:
            self.failed = "not enough samples - keeping defaults"
            return

        # Trimmed ends rather than min/max, so one bad landmark frame cannot
        # define the whole range.
        lo = statistics.median(sorted(down)[: max(1, len(down) // 2)])
        hi = statistics.median(sorted(up)[-max(1, len(up) // 2) :])
        rng = hi - lo

        if rng < MIN_RANGE:
            self.failed = f"arms-up and arms-down look the same (range {rng:.2f})"
            return

        # In adaptive mode the band follows recent motion, so calibration's job
        # is to scale "how much motion counts as a flap" to this body, and to
        # leave sane fixed values behind for --fixed-thresholds.
        s = self.sens
        cfg.apply_sensitivity(self.conf, s)
        # Scale the 'is this a flap' floor to this body, but never below the
        # preset's own floor, which is what keeps jitter out.
        self.conf.min_span = max(s.min_span, 0.11 * rng)
        self.conf.h_up = lo + s.up_frac * rng
        self.conf.h_down = lo + s.down_frac * rng
        self.conf.v_thresh = min(2.5, max(0.5, s.v_frac * rng * 0.5))
        self.summary = (
            f"range {lo:+.2f}..{hi:+.2f}  min_span {self.conf.min_span:.2f}  [{s.name}]"
        )

    def restart(self) -> None:
        self.phase = Phase.DOWN
        self.elapsed = 0.0
        self.samples = {Phase.DOWN: [], Phase.UP: []}
        self.failed = None
