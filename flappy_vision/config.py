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


# Two separate levers: hover_rate decides how *tiring* a preset is,
# pipe_gap/pipe_spacing decide how *precise* you have to be. The bird's
# collision box is only 36px tall, so normal's 400px gap in a 624px playfield
# leaves roughly 10x the bird's height of slack.
#
# The ladder is anchored so that `normal` is the comfortable arm-flapping
# experience rather than a challenge - hovering costs under one flap per second.
# `hard` and `classic` are where the original game's cruelty lives.
# Scoring pace is pipe_spacing / pipe_speed. Wide gaps buy a lot of slack, and
# the right thing to spend it on is tempo: a point every ~2s feels like a game,
# a point every ~5s feels like waiting. Speed is raised and spacing tightened
# together so pipes arrive often without the scroll becoming a blur.
CHILL = Difficulty("chill", 650.0, -430.0, 460, 120.0, 300)
NORMAL = Difficulty("normal", 780.0, -430.0, 400, 140.0, 300)
HARD = Difficulty("hard", 1400.0, -430.0, 280, 175.0, 330)
CLASSIC = Difficulty("classic", 2100.0, -520.0, 200, 215.0, 320)

# Order matters: this is the order shown in the in-game selector.
DIFFICULTY_ORDER = (CHILL, NORMAL, HARD, CLASSIC)
DIFFICULTIES = {d.name: d for d in DIFFICULTY_ORDER}
DEFAULT_DIFFICULTY = "normal"


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


# min_span, not up_frac, is what decides how small a flap can be. Swept in
# scripts/verify_detector.py: dropping up_frac 0.62 -> 0.42 does not improve
# small-flap detection at all, while dropping min_span 0.30 -> 0.22 takes an
# amplitude-0.55 flap from 3/10 to 10/10.
#
# The cost is false fires. Still arms with noisy tracking produce an envelope up
# to ~0.12 wide (measured at sigma 0.08), so the floor cannot go arbitrarily low.
# The presets therefore differ almost only in min_span. Moving up_frac/v_frac
# alongside it measurably *hurt*: a "high" preset that also lowered those scored
# 9/10 on amplitude-0.55 flaps where one changing min_span alone scored 10/10.
SENS_LOW = Sensitivity("low", 0.55, 0.32, 0.45, 0.34)  # big deliberate flaps only
SENS_NORMAL = Sensitivity("normal", 0.55, 0.32, 0.35, 0.22)  # ordinary wing flap
SENS_HIGH = Sensitivity("high", 0.55, 0.32, 0.28, 0.16)  # tiny flaps; may false-fire

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
    up_frac: float = 0.55  # h_up position within the recent envelope
    down_frac: float = 0.32  # h_down position
    min_span: float = 0.22  # envelope narrower than this = "not flapping"
    v_frac: float = 0.35  # velocity gate as a fraction of envelope span
    v_floor: float = 0.40  # never gate below this, or noise fires it

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
