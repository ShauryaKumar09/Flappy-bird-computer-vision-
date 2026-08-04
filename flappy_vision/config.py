"""All tunable constants live here.

Physics is expressed in pixels-per-second (not per-frame) and integrated against a
real dt, so the bird behaves identically whether the render loop runs at 20 or 120 FPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
SCORE_FILE = ROOT / ".best_score"

# ---------------------------------------------------------------- layout ----
# ┌──────────────┬──────────────────────┐
# │  GAME        │  CAMERA              │
# │  512x720     │  768x576             │
# │              ├──────────────────────┤
# │              │  HUD 768x144         │
# └──────────────┴──────────────────────┘
GAME_W, GAME_H = 512, 720
CAM_W, CAM_H = 768, 576
HUD_H = GAME_H - CAM_H  # 144
WINDOW_W, WINDOW_H = GAME_W + CAM_W, GAME_H

# base.png is 336x112; keeping that 3:1 aspect means the ground tile is 3*GROUND_H
# wide, and the strip is tiled rather than stretched.
GROUND_H = 96
GROUND_TILE_W = GROUND_H * 3
PLAY_H = GAME_H - GROUND_H  # 624 usable vertical pixels

BIRD_X = GAME_W // 3
BIRD_W, BIRD_H = 68, 48
PIPE_W = 88
PIPE_SPRITE_H = 640

# Physics is authored against this height; presets scale if PLAY_H changes.
TERMINAL_VELOCITY = 700.0  # px/s, downward clamp
MAX_DT = 1 / 20  # clamp so a stall can't tunnel the bird through a pipe


@dataclass(frozen=True)
class Difficulty:
    name: str
    gravity: float  # px/s^2
    flap_impulse: float  # px/s, negative = upward
    pipe_gap: int  # px of vertical opening
    pipe_speed: float  # px/s leftward
    pipe_spacing: int  # px between consecutive pipes (distance-based spawning)

    @property
    def hover_rate(self) -> float:
        """Flaps per second needed to hold altitude - sanity-checks a preset."""
        return self.gravity / (2 * abs(self.flap_impulse))


# hover_rate is the number that decides how tiring a preset is, not pipe_gap.
# chill/easy sit near 1 flap/sec, which is a sustainable arm rhythm.
CHILL = Difficulty("chill", 780.0, -430.0, 300, 100.0, 400)
EASY = Difficulty("easy", 950.0, -430.0, 275, 118.0, 375)
NORMAL = Difficulty("normal", 1400.0, -430.0, 220, 150.0, 340)
CLASSIC = Difficulty("classic", 2100.0, -520.0, 160, 210.0, 300)

DIFFICULTIES = {d.name: d for d in (CHILL, EASY, NORMAL, CLASSIC)}


# -------------------------------------------------------------- detector ----


@dataclass(frozen=True)
class Sensitivity:
    """How much of your own flap you have to complete before it counts.

    These are fractions of the *recent motion envelope*, not absolute heights,
    so they describe gesture shape rather than arm position. "high" fires
    partway through a stroke; "low" wants a fuller committed swing.
    """

    name: str
    up_frac: float  # arm the trigger this far up the envelope
    down_frac: float  # re-arm below this far up the envelope
    v_frac: float  # velocity gate as a fraction of envelope span
    min_span: float  # smallest flap that counts, in shoulder-widths


# min_span is the real tradeoff. Still arms with noisy tracking produce an
# envelope up to ~0.12 wide (measured, sigma 0.08), so dropping the floor to
# catch tiny flaps eventually starts catching jitter. 0.55 shoulder-widths is
# roughly 22cm of wrist travel - a genuinely small flap - and is safe at 0.30.
SENS_LOW = Sensitivity("low", 0.70, 0.45, 0.60, 0.40)  # big deliberate flaps only
SENS_NORMAL = Sensitivity("normal", 0.62, 0.38, 0.45, 0.30)  # ordinary wing flap
SENS_HIGH = Sensitivity("high", 0.55, 0.32, 0.32, 0.20)  # tiny flaps; may false-fire

SENSITIVITIES = {s.name: s for s in (SENS_LOW, SENS_NORMAL, SENS_HIGH)}


def apply_sensitivity(conf: "DetectorConfig", sens: Sensitivity) -> None:
    conf.up_frac = sens.up_frac
    conf.down_frac = sens.down_frac
    conf.v_frac = sens.v_frac
    conf.min_span = sens.min_span


@dataclass
class DetectorConfig:
    """Thresholds for the flap state machine, in shoulder-width units.

    Defaults suit an average build flapping near shoulder height; calibration
    replaces them with values derived from the player's own range.
    """

    # h_down is the *re-arm* threshold, so setting it too low is a trap: at fast
    # flap rates the trough is narrow and the smoothing rounds it off, so the
    # signal never dips under it and the detector latches after one flap. Keep a
    # wide hysteresis band, but not a strict one.
    h_up: float = 0.14  # arm elevation that arms the trigger
    h_down: float = -0.14  # must fall below this to re-arm (hysteresis)
    v_thresh: float = 1.0  # downstroke speed (shoulder-widths/sec) that fires
    refractory_s: float = 0.22  # minimum time between flaps
    mode: str = "flap"  # "flap" (velocity) or "raise" (static threshold)

    # Adaptive mode tracks a decaying envelope of recent elevation and places
    # the band inside it, so where you hold your arms stops mattering. The
    # fixed h_up/h_down above are the fallback when this is off.
    adaptive: bool = True
    up_frac: float = 0.62  # h_up position within the recent envelope
    down_frac: float = 0.38  # h_down position
    min_span: float = 0.30  # envelope narrower than this = "not flapping"
    v_frac: float = 0.45  # velocity gate as a fraction of envelope span
    v_floor: float = 0.55  # never gate below this, or noise fires it

    # One-euro filter tuning for the elevation signal.
    min_cutoff: float = 1.2
    beta: float = 0.5


# --------------------------------------------------------------- colors ----
# BGR, since OpenCV.
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_PANEL = (28, 24, 20)
C_ACCENT = (80, 220, 255)  # amber-cyan
C_GOOD = (100, 255, 100)
C_WARN = (80, 180, 255)
C_BAD = (80, 80, 255)
C_DIM = (110, 100, 95)
