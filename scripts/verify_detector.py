"""Headless checks for the flap detector.

Synthetic arm-elevation signals stand in for a person flapping, which makes the
"no double-fires, no misses" criterion deterministic and repeatable instead of
something you can only test by flailing at a webcam.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from flappy_vision import config as cfg
from flappy_vision.pose import FlapDetector, arm_elevation

FPS = 30.0

# Flap styles as (trough, peak) arm elevation in shoulder-widths, described
# physically rather than relative to any threshold. "small" is the one that
# matters: a tired player doing little wing flaps around shoulder height.
STYLES = {
    "full": (-1.20, 1.60),  # thigh to overhead
    "medium": (-0.60, 0.90),  # below horizontal to above the head-line
    "small": (-0.35, 0.35),  # small flaps centred on the shoulders
}


def flap_signal(
    n_flaps: int, hz: float, noise: float, style: str = "full", seed: int = 0
) -> list[tuple[float, float]]:
    """One raised-cosine cycle per flap: arms up, then down. Returns [(t, H)]."""
    rng = np.random.default_rng(seed)
    lo, hi = STYLES[style]
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    n = int((n_flaps / hz) * FPS)
    out = []
    for i in range(n):
        t = i / FPS
        h = mid - amp * math.cos(2 * math.pi * hz * t)
        out.append((t, h + rng.normal(0, noise)))
    return out


def count_flaps(samples, conf: cfg.DetectorConfig) -> int:
    det = FlapDetector(conf)
    return sum(det.update(h, t) for t, h in samples)


def check_rates() -> bool:
    """Hovering costs ~1.6 flaps/sec, so the mid-range rates are the ones that matter."""
    ok = True
    conf = cfg.DetectorConfig()
    for style in STYLES:
        lo, hi = STYLES[style]
        print(f"\nflap rate sweep [{style}: H {lo:+.2f}..{hi:+.2f}], noise sigma=0.03:")
        for hz in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
            n = 10
            got = count_flaps(flap_signal(n, hz, noise=0.03, style=style), conf)
            good = got == n
            ok &= good
            peak_v = (hi - lo) / 2 * 2 * math.pi * hz
            print(
                f"  {hz:4.2f} Hz  expected {n:3d}  got {got:3d}  "
                f"peak |dH/dt| {peak_v:5.2f}/s   {'ok' if good else 'MISMATCH'}"
            )
    return ok


def check_no_chatter() -> bool:
    """Arms parked right on the threshold - the case that breaks naive detectors.

    A bare `wrist.y < shoulder.y` edge test fires every time jitter crosses the
    line. With hysteresis plus a refractory window this must stay near-silent.
    """
    conf = cfg.DetectorConfig()
    ok = True
    parks = {
        "at h_up": conf.h_up,
        "mid-band": (conf.h_up + conf.h_down) / 2,
        "at h_down": conf.h_down,
    }
    for where, level in parks.items():
        print(f"\nchatter test (arms parked {where} = {level:.2f}, 10s of jitter):")
        for sigma in (0.02, 0.05, 0.10):
            rng = np.random.default_rng(3)
            samples = [(i / FPS, level + rng.normal(0, sigma)) for i in range(int(10 * FPS))]
            got = count_flaps(samples, conf)
            good = got <= 1
            ok &= good
            print(f"  sigma {sigma:4.2f}  spurious flaps {got:3d}   {'ok' if good else 'CHATTER'}")
    return ok


def check_naive_baseline() -> None:
    """Show what the reference implementation's approach would do on the same input."""
    rng = np.random.default_rng(3)
    samples = [(i / FPS, 0.0 + rng.normal(0, 0.05)) for i in range(int(10 * FPS))]
    # Reference logic: raised = wrist above shoulder (H > 0), fire on rising edge.
    fired, was = 0, False
    for _, h in samples:
        now = h > 0.0
        fired += now and not was
        was = now
    print(f"\n  for contrast, a bare H>0 edge test on the same jitter: {fired} spurious flaps")


@dataclass
class FakeLM:
    x: float
    y: float
    visibility: float = 1.0


def check_scale_invariance() -> bool:
    """The same gesture at different camera distances must give the same H."""
    print("\nscale invariance (identical pose, three apparent body sizes):")
    values = []
    for scale in (0.10, 0.20, 0.35):
        lms = [FakeLM(0, 0) for _ in range(33)]
        lms[11] = FakeLM(0.5 - scale / 2, 0.5)  # left shoulder
        lms[12] = FakeLM(0.5 + scale / 2, 0.5)  # right shoulder
        lms[15] = FakeLM(0.5 - scale / 2, 0.5 - scale * 1.2)  # left wrist, raised
        lms[16] = FakeLM(0.5 + scale / 2, 0.5)  # right wrist, down
        h = arm_elevation(lms)
        values.append(h)
        print(f"  shoulder width {scale:.2f} of frame -> H = {h:.4f}")
    spread = max(values) - min(values)
    print(f"  spread {spread:.2e} (should be ~0)")
    return spread < 1e-9


def check_low_visibility() -> bool:
    lms = [FakeLM(0.5, 0.5, visibility=0.9) for _ in range(33)]
    lms[15] = FakeLM(0.4, 0.2, visibility=0.1)  # wrist not confidently seen
    got = arm_elevation(lms)
    print(f"\nlow-visibility wrist -> {got} (expect None, so state holds)")
    return got is None


@dataclass
class FakeSnap:
    h: float
    have_pose: bool = True


def _calibrate(lo: float, hi: float, sens: cfg.Sensitivity) -> cfg.DetectorConfig:
    from flappy_vision.calibrate import COLLECT_S, SETTLE_S, Calibrator

    conf = cfg.DetectorConfig()
    cal = Calibrator(conf, sens)
    rng = np.random.default_rng(11)
    dt = 1 / FPS
    for level in (lo, hi):
        for _ in range(int((SETTLE_S + COLLECT_S) / dt) + 2):
            cal.update(FakeSnap(level + rng.normal(0, 0.03)), dt)
    assert cal.done and not cal.failed, cal.failed
    return conf


def _flaps_at(conf, lo: float, hi: float, hz: float, n: int = 10) -> int:
    rng = np.random.default_rng(5)
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    samples = [
        (
            i / FPS,
            mid - amp * math.cos(2 * math.pi * hz * (i / FPS)) + rng.normal(0, 0.03),
        )
        for i in range(int((n / hz) * FPS))
    ]
    return count_flaps(samples, conf)


def check_position_invariance() -> bool:
    """Same flap, different arm heights. All must count the same.

    This is the failure that makes the game feel broken: you drift out of a
    fixed band as you tire, and the trigger silently stops firing.
    """
    print("\nposition invariance (same 0.9-wide flap, different arm heights):")
    conf = cfg.DetectorConfig()
    ok = True
    for centre in (-1.20, -0.60, 0.00, 0.60, 1.20):
        got = [_flaps_at(conf, centre - 0.45, centre + 0.45, hz) for hz in (0.75, 1.5, 2.5)]
        good = all(g == 10 for g in got)
        ok &= good
        print(f"  centre {centre:+.2f}  counts {got}   {'ok' if good else 'MISSES'}")
    return ok


def check_amplitude_invariance() -> bool:
    """Big committed flaps and small tired ones must both count.

    Each preset has a documented smallest reliable flap; below it, detection is
    expected to fall off. That floor is the honest answer to "how small a flap
    can I get away with", and `-s high` is what to reach for when tired.
    """
    print("\namplitude coverage (centre -0.3, min count across 0.75/1.5/2.5 Hz):")
    floors = {"low": 1.60, "normal": 0.90, "high": 0.55}
    ok = True
    for sens in (cfg.SENS_LOW, cfg.SENS_NORMAL, cfg.SENS_HIGH):
        conf = cfg.DetectorConfig()
        cfg.apply_sensitivity(conf, sens)
        row = []
        for amp in (2.60, 1.60, 0.90, 0.55, 0.35):
            worst = min(
                _flaps_at(conf, -0.3 - amp / 2, -0.3 + amp / 2, hz) for hz in (0.75, 1.5, 2.5)
            )
            row.append(worst)
            if amp >= floors[sens.name] and worst != 10:
                ok = False
        print(f"  {sens.name:<6} amp 2.60/1.60/0.90/0.55/0.35 -> {row}   floor {floors[sens.name]}")
    return ok


def check_still_arms() -> bool:
    """Holding arms anywhere, however jittery, must not fire.

    This is what the min_span guard buys: the envelope collapses when you stop
    moving, and a collapsed envelope is not a flap.
    """
    print("\nstill arms (min_span guard, 10s each):")
    conf = cfg.DetectorConfig()
    ok = True
    for level in (-1.2, 0.0, 1.2):
        for sigma in (0.03, 0.08):
            rng = np.random.default_rng(3)
            samples = [(i / FPS, level + rng.normal(0, sigma)) for i in range(int(10 * FPS))]
            got = count_flaps(samples, conf)
            good = got == 0
            ok &= good
            print(f"  held at {level:+.1f} sigma {sigma:.2f} -> {got:3d} flaps  "
                  f"{'ok' if good else 'FALSE FIRE'}")
    return ok


def check_sensitivity_ordering() -> bool:
    """high must fire on at least as much as normal, and normal on at least low."""
    print("\nsensitivity presets (partial flaps - stroke cut short):")
    counts = {}
    for sens in (cfg.SENS_LOW, cfg.SENS_NORMAL, cfg.SENS_HIGH):
        conf = cfg.DetectorConfig()
        cfg.apply_sensitivity(conf, sens)
        # A player who only completes part of each stroke: the envelope stays
        # wide from earlier big flaps, but current motion covers less of it.
        rng = np.random.default_rng(9)
        samples = []
        for i in range(int(12 * FPS)):
            t = i / FPS
            big = 1.3 if (t % 4.0) < 1.5 else 0.55  # big flaps, then partial ones
            samples.append((t, -0.3 - big / 2 * math.cos(2 * math.pi * 1.5 * t) + rng.normal(0, 0.03)))
        counts[sens.name] = count_flaps(samples, conf)
    print(f"  {counts}")
    ok = counts["high"] >= counts["normal"] >= counts["low"]
    print(f"  ordering high >= normal >= low: {ok}")
    return ok


def check_calibration_rejects_garbage() -> bool:
    from flappy_vision.calibrate import COLLECT_S, SETTLE_S, Calibrator

    conf = cfg.DetectorConfig()
    before = (conf.h_up, conf.h_down)
    cal = Calibrator(conf)
    dt = 1 / FPS
    # Player never moves: arms-up and arms-down look identical.
    for _ in range(2 * (int((SETTLE_S + COLLECT_S) / dt) + 2)):
        cal.update(FakeSnap(0.5), dt)
    kept = (conf.h_up, conf.h_down) == before
    print(f"\nmotionless player -> rejected: {cal.failed!r}, defaults kept: {kept}")
    return kept and cal.failed is not None


if __name__ == "__main__":
    results = {
        "rate sweep": check_rates(),
        "no chatter": check_no_chatter(),
        "scale invariance": check_scale_invariance(),
        "visibility gate": check_low_visibility(),
        "position invariance": check_position_invariance(),
        "amplitude invariance": check_amplitude_invariance(),
        "still arms": check_still_arms(),
        "sensitivity ordering": check_sensitivity_ordering(),
        "calibration guard": check_calibration_rejects_garbage(),
    }
    check_naive_baseline()
    print()
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    raise SystemExit(0 if all(results.values()) else 1)
