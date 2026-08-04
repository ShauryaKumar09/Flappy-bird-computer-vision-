"""Drive the whole app state machine headlessly.

Mirrors main.py's loop body but without cv2 windowing, so the calibrate -> play
-> die -> restart path is exercised in CI-ish conditions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from flappy_vision import config as cfg
from flappy_vision.autopilot import Autopilot
from flappy_vision.calibrate import COLLECT_S, SETTLE_S, Calibrator
from flappy_vision.effects import Effects
from flappy_vision.game import GameState, World
from flappy_vision.pose import ARM_POINTS, FlapState, Snapshot
from flappy_vision.render import Assets, render_calibration, render_camera, render_game, render_hud

DT = 1 / 60


@dataclass
class Fake:
    h: float
    have_pose: bool = True
    state: FlapState = FlapState.DOWN
    v: float = 0.0
    infer_ms: float = 8.0
    vision_fps: float = 30.0
    h_up: float = 0.14
    h_down: float = -0.14
    span: float = 0.9
    active: bool = True
    points: dict = None
    frame: np.ndarray = None

    def __post_init__(self):
        if self.frame is None:
            self.frame = np.zeros((480, 640, 3), np.uint8)
        if self.points is None:
            self.points = {i: (0.5, 0.5) for i in ARM_POINTS}


def main() -> int:
    assets = Assets()
    conf = cfg.DetectorConfig()
    world = World(diff=cfg.NORMAL)
    world.rng.seed(3)
    effects = Effects()
    cal = Calibrator(conf)
    checks = {}

    # --- calibrate -------------------------------------------------------
    steps = int((SETTLE_S + COLLECT_S) / DT) + 4
    for level in (-0.85, 1.45):
        for _ in range(steps):
            snap = Fake(level)
            if not cal.done:
                cal.update(snap, DT)
                render_calibration(render_game(world, assets, effects, chrome=False), cal)
    checks["calibration completes"] = cal.done and cal.failed is None
    checks["thresholds moved"] = (conf.h_up, conf.h_down) != (0.70, 0.45)
    print(f"calibrated: up {conf.h_up:.2f} down {conf.h_down:.2f} v {conf.v_thresh:.2f}")

    # --- play with the autopilot standing in for a player -----------------
    bot = Autopilot()
    t, deaths, bursts = 0.0, 0, 0
    while t < 25.0:
        snap = Fake(0.9, state=FlapState.ARMED)
        if bot.step(world, DT):
            world.flap()
        world.update(DT)
        if world.just_died:
            deaths += 1
            effects.burst(cfg.BIRD_X, world.bird.y)
            bursts = len(effects.particles)
        effects.update(DT)

        frame = np.hstack(
            (
                render_game(world, assets, effects),
                np.vstack(
                    (
                        render_camera(snap, conf),
                        render_hud(snap, conf, {"render_fps": 60, "flaps": 3}),
                    )
                ),
            )
        )
        if t == 0.0:
            checks["composite is 1280x720"] = frame.shape == (720, 1280, 3)
        t += DT

    print(f"played 25s: score {world.score}, state {world.state.name}")
    checks["game advanced"] = world.score > 0

    # --- forced death, effects, restart ----------------------------------
    world.reset()
    world.bird.y = cfg.PLAY_H - 5  # drop it on the floor
    for _ in range(30):
        world.update(DT)
        if world.just_died:
            effects.burst(cfg.BIRD_X, world.bird.y)
    checks["death detected"] = world.state is GameState.GAME_OVER
    checks["particles spawned"] = len(effects.particles) > 0

    for _ in range(120):  # particles should expire, shake should settle
        effects.update(DT)
    checks["effects settle"] = effects.shake_t == 0.0

    world.flap()  # flap-to-retry
    checks["restart works"] = world.state is GameState.PLAYING

    # --- best score persistence ------------------------------------------
    from flappy_vision.main import load_best, save_best

    original = load_best()
    save_best(4242)
    checks["best score persists"] = load_best() == 4242
    save_best(original)

    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
