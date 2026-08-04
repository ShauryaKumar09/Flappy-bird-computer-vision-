"""Render still frames headlessly, for eyeballing the look without playing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from flappy_vision import config as cfg
from flappy_vision.autopilot import Autopilot
from flappy_vision.calibrate import Calibrator
from flappy_vision.effects import Effects
from flappy_vision.game import GameState, World
from flappy_vision.pose import ARM_POINTS, FlapState, Snapshot
from flappy_vision.render import (
    Assets,
    render_calibration,
    render_camera,
    render_game,
    render_hud,
)

OUT = Path("/tmp")


def play_for(seconds: float, diff: cfg.Difficulty = cfg.NORMAL, seed: int = 7) -> World:
    w = World(diff=diff)
    w.rng.seed(seed)
    w.reset()
    bot, dt, t = Autopilot(), 1 / 60, 0.0
    while t < seconds and w.state is GameState.PLAYING:
        if bot.step(w, dt):
            w.bird.flap(w.diff)
        w.update(dt)
        t += dt
    return w


def fake_snapshot(h: float, state: FlapState) -> Snapshot:
    """A stand-in webcam frame with a plausible arm pose, so the HUD can be seen."""
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :] = (60, 48, 40)
    cv2.rectangle(img, (0, 300), (640, 480), (78, 62, 52), -1)

    # Shoulders at mid-frame; wrists raised by `h` shoulder-widths.
    sw = 0.22
    pts = {
        11: (0.5 - sw / 2, 0.45),
        12: (0.5 + sw / 2, 0.45),
        13: (0.5 - sw / 2 - 0.06, 0.45 - h * sw * 0.5),
        14: (0.5 + sw / 2 + 0.06, 0.45 - h * sw * 0.5),
        15: (0.5 - sw / 2 - 0.10, 0.45 - h * sw),
        16: (0.5 + sw / 2 + 0.10, 0.45 - h * sw),
    }
    return Snapshot(
        frame=img,
        points={i: pts[i] for i in ARM_POINTS},
        have_pose=True,
        h=h,
        v=-3.2,
        state=state,
        infer_ms=8.0,
        vision_fps=30.1,
    )


def compose(game: np.ndarray, snap: Snapshot, conf: cfg.DetectorConfig, stats: dict) -> np.ndarray:
    return np.hstack((game, np.vstack((render_camera(snap, conf), render_hud(snap, conf, stats)))))


def main() -> None:
    assets = Assets()
    conf = cfg.DetectorConfig()
    stats = {
        "render_fps": 58.9,
        "vision_fps": 30.1,
        "infer_ms": 8.0,
        "latency_ms": 74.3,
        "flaps": 37,
    }

    # 1. mid-play, arms up and armed
    playing = play_for(14.0)
    shot_play = compose(
        render_game(playing, assets), fake_snapshot(1.15, FlapState.ARMED), conf, stats
    )

    # 2. calibration prompt
    fresh = World(diff=cfg.NORMAL)
    cal = Calibrator(cfg.DetectorConfig())
    cal.elapsed = 2.4  # partway through sampling, so the bar is filled
    shot_cal = compose(
        render_calibration(render_game(fresh, assets, chrome=False), cal),
        fake_snapshot(-0.7, FlapState.DOWN),
        conf,
        stats,
    )

    # 3. the moment of death: particles mid-flight plus screen shake
    dead = play_for(14.0)
    dead.state = GameState.GAME_OVER
    dead.best = 12
    fx = Effects()
    fx.rng.seed(4)
    fx.burst(cfg.BIRD_X, dead.bird.y)
    for _ in range(14):
        fx.update(1 / 60)
    shot_dead = compose(render_game(dead, assets, fx), fake_snapshot(0.2, FlapState.RECOVER), conf, stats)

    for name, img in (("play", shot_play), ("calibrate", shot_cal), ("death", shot_dead)):
        path = OUT / f"flappy_{name}.png"
        cv2.imwrite(str(path), img)
        print(f"wrote {path}  {img.shape}")


if __name__ == "__main__":
    main()
