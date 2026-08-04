"""Headless checks for the game simulation.

The important one is dt-independence: the reference implementation this project is
based on used per-frame velocity units, so its bird physics silently changed
whenever the framerate dropped. These assertions prove ours doesn't.
"""

from __future__ import annotations

import statistics
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from flappy_vision import config as cfg
from flappy_vision.autopilot import Autopilot
from flappy_vision.game import GameState, World


def trajectory(fps: float, duration: float, flap_times: list[float]) -> list[float]:
    """Run the sim at a fixed framerate, flapping at the given wall-clock times."""
    w = World(diff=cfg.NORMAL)
    w.rng.seed(1234)
    w.reset()
    dt = 1.0 / fps
    pending = sorted(flap_times)
    ys, t = [], 0.0
    while t < duration:
        # Record before stepping, so ys[i] is the position at exactly t = i*dt.
        # Recording after the step offsets each series by its own dt, which shows
        # up as fake dt-dependence.
        ys.append(w.bird.y)
        while pending and pending[0] <= t + 1e-9:
            pending.pop(0)
            w.bird.flap(w.diff)
        w.bird.update(dt, w.diff)
        t += dt
    return ys


def sample(ys: list[float], fps: float, at: list[float]) -> list[float]:
    return [ys[min(len(ys) - 1, round(t * fps))] for t in at]


def check_dt_independence() -> bool:
    # Flap and probe times are multiples of 0.1s, which lands exactly on a frame
    # boundary for every framerate tested. That isolates integration error from
    # input quantisation - a flap arriving a frame late is a real effect, but not
    # the thing this check is about.
    flaps = [0.2, 0.6, 1.0, 1.5, 2.0]
    probes = [0.5, 1.0, 1.5, 2.0, 2.4]
    ref = sample(trajectory(120, 2.5, flaps), 120, probes)

    ok = True
    print(f"{'fps':>6} " + " ".join(f"t={t:<5.1f}" for t in probes) + "   max err")
    print(f"{120:>6} " + " ".join(f"{y:7.1f}" for y in ref) + "        --")
    for fps in (20, 30, 60, 90):
        ys = sample(trajectory(fps, 2.5, flaps), fps, probes)
        err = max(abs(a - b) for a, b in zip(ys, ref))
        print(f"{fps:>6} " + " ".join(f"{y:7.1f}" for y in ys) + f"   {err:7.2f}px")
        if err > 1.0:
            ok = False
    return ok


def check_hover_rates() -> None:
    print("\nflaps/sec needed to hold altitude:")
    for d in (cfg.CHILL, cfg.EASY, cfg.NORMAL, cfg.CLASSIC):
        rise = d.flap_impulse**2 / (2 * d.gravity)
        print(f"  {d.name:<8} {d.hover_rate:4.2f}/s   apex rise {rise:5.1f}px   gap {d.pipe_gap}px")


def check_gameplay(diff: cfg.Difficulty, floor: int, seeds: int = 6) -> bool:
    """Median autopilot score over several seeds must clear `floor`.

    A single seed is far too noisy to judge a difficulty preset on - one unlucky
    pipe placement early and the run ends at 0.
    """
    runs = [_run_one(diff, s) for s in range(seeds)]
    scores = [r[0] for r in runs]
    survived = sum(1 for r in runs if not r[1])
    med = statistics.median(scores)
    print(
        f"  {diff.name:<8} scores={scores}  median={med:.1f}  "
        f"survived {survived}/{seeds}  (floor {floor})"
    )
    return med >= floor


def _run_one(diff: cfg.Difficulty, seed: int, cap_s: float = 60.0) -> tuple[int, bool]:
    """One autopilot run -> (score, died). A proxy for 'a competent player can play this'."""
    w = World(diff=diff)
    w.rng.seed(seed)
    w.reset()
    bot = Autopilot()
    dt, t = 1 / 60, 0.0

    while t < cap_s and w.state is GameState.PLAYING:
        if bot.step(w, dt):
            w.bird.flap(w.diff)
        w.update(dt)
        t += dt
    return w.score, w.state is GameState.GAME_OVER


def check_collision() -> bool:
    w = World(diff=cfg.NORMAL)
    w.reset()
    for _ in range(600):  # never flap -> must hit the ground
        w.update(1 / 60)
        if w.state is GameState.GAME_OVER:
            print(f"free-fall death at y={w.bird.y:.0f} (play height {cfg.PLAY_H})")
            return True
    return False


if __name__ == "__main__":
    results = {
        "dt-independence": check_dt_independence(),
        "collision": check_collision(),
    }
    # Classic is meant to be brutal, so it gets a lower bar - the check is that
    # it stays *playable*, not that it is easy.
    print("\nautopilot (60s cap, 6 seeds):")
    for d, floor in ((cfg.CHILL, 12), (cfg.EASY, 14), (cfg.NORMAL, 12), (cfg.CLASSIC, 5)):
        results[f"autopilot/{d.name}"] = check_gameplay(d, floor)
    check_hover_rates()
    print()
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    raise SystemExit(0 if all(results.values()) else 1)
