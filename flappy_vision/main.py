"""Entry point: CLI, window, main loop."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg
from .autopilot import Autopilot
from .calibrate import Calibrator
from .capture import CameraError, list_cameras
from .effects import Effects
from .game import GameState, World
from .render import (
    FONT,
    Assets,
    render_calibration,
    render_camera,
    render_game,
    render_hud,
)

WINDOW = "Flappy Vision"

KEY_QUIT = {27, ord("q")}
KEY_FLAP = {32}
KEY_RESET = {ord("r")}
KEY_CALIBRATE = {ord("c")}
KEY_AUTOPILOT = {ord("a")}
# 1-4 pick a difficulty, in the order shown by the on-screen picker.
KEY_DIFFICULTY = {ord(str(i + 1)): d for i, d in enumerate(cfg.DIFFICULTY_ORDER)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flappy", description=__doc__)
    p.add_argument(
        "-d",
        "--difficulty",
        choices=[d.name for d in cfg.DIFFICULTY_ORDER],
        default=cfg.DEFAULT_DIFFICULTY,
        help="starting difficulty; also switchable in-game with 1-4",
    )
    p.add_argument("--keyboard", action="store_true", help="spacebar only; skip the camera")
    p.add_argument("-c", "--camera", type=int, default=0, help="camera index")
    p.add_argument("--list-cameras", action="store_true", help="show available cameras and exit")
    p.add_argument(
        "--mode",
        choices=("flap", "raise"),
        default="flap",
        help="'flap' fires on the arm downstroke; 'raise' on a static arm-up threshold",
    )
    p.add_argument(
        "-s",
        "--sensitivity",
        choices=sorted(cfg.SENSITIVITIES),
        default="normal",
        help="how big a flap has to be: 'high' accepts small tired flaps",
    )
    p.add_argument(
        "--fixed-thresholds",
        action="store_true",
        help="disable the adaptive band and use absolute thresholds",
    )
    p.add_argument("--no-calibrate", action="store_true", help="use default thresholds")
    p.add_argument("--fps", type=int, default=60, help="render loop target FPS")
    p.add_argument("--seed", type=int, default=None, help="fix pipe layout for testing")
    p.add_argument("--record", type=Path, default=None, help="write a landmark trace to JSON")
    return p


def load_best() -> int:
    try:
        return int(cfg.SCORE_FILE.read_text().strip())
    except (OSError, ValueError):
        return 0


def save_best(score: int) -> None:
    try:
        cfg.SCORE_FILE.write_text(str(score))
    except OSError:
        pass  # a read-only home directory should not crash the game


def placeholder_pane(text: str, h: int = cfg.GAME_H) -> np.ndarray:
    pane = np.full((h, cfg.CAM_W, 3), cfg.C_PANEL, np.uint8)
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.7, 1)
    cv2.putText(pane, text, ((cfg.CAM_W - tw) // 2, h // 2), FONT, 0.7, cfg.C_DIM, 1, cv2.LINE_AA)
    return pane


def run(args: argparse.Namespace) -> int:
    assets = Assets()
    world = World(diff=cfg.DIFFICULTIES[args.difficulty])
    world.best = load_best()
    if args.seed is not None:
        world.rng.seed(args.seed)

    conf = cfg.DetectorConfig(mode=args.mode, adaptive=not args.fixed_thresholds)
    cfg.apply_sensitivity(conf, cfg.SENSITIVITIES[args.sensitivity])
    effects = Effects()

    vision = None
    if not args.keyboard:
        from .pose import VisionThread

        vision = VisionThread(camera=args.camera, conf=conf, record=args.record is not None)
        vision.start()

    sens = cfg.SENSITIVITIES[args.sensitivity]
    cal = Calibrator(conf, sens) if (vision is not None and not args.no_calibrate) else None
    bot: Autopilot | None = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    target_dt = 1.0 / max(1, args.fps)
    prev = time.perf_counter()
    render_fps = float(args.fps)
    latency_ms = 0.0
    flaps = 0
    idle_pane = placeholder_pane(
        "keyboard mode - press SPACE to flap" if vision is None else "waiting for camera...",
        cfg.CAM_H,
    )

    while True:
        now = time.perf_counter()
        dt = now - prev
        prev = now
        render_fps = 0.9 * render_fps + 0.1 / max(dt, 1e-6)

        snap = None
        if vision is not None:
            if vision.error is not None:
                raise vision.error
            snap = vision.latest()

        calibrating = cal is not None and not cal.done
        if calibrating:
            cal.update(snap, dt)
            if vision is not None:
                vision.drain_flaps()  # discard; calibration is not gameplay
        elif vision is not None:
            for ev in vision.drain_flaps():
                world.flap()
                flaps += 1
                # Measured from frame capture, not from detection - that is the
                # number describing how late the bird actually responds.
                latency_ms = 0.7 * latency_ms + 0.3 * (time.perf_counter() - ev.t_capture) * 1000

        if bot is not None and not calibrating and bot.step(world, dt):
            world.flap()

        if not calibrating:
            world.update(dt)
            if world.just_died:
                effects.burst(cfg.BIRD_X, world.bird.y)
                save_best(world.best)
        effects.update(dt)

        game = render_game(world, assets, effects, chrome=not calibrating)
        if calibrating:
            game = render_calibration(game, cal)

        top = render_camera(snap, conf) if snap is not None else idle_pane
        stats = {
            "render_fps": render_fps,
            "vision_fps": snap.vision_fps if snap else 0.0,
            "infer_ms": snap.infer_ms if snap else 0.0,
            "latency_ms": latency_ms,
            "flaps": flaps,
        }
        frame = np.hstack((game, np.vstack((top, render_hud(snap, conf, stats)))))
        cv2.imshow(WINDOW, frame)

        spent = time.perf_counter() - now
        key = cv2.waitKey(max(1, int((target_dt - spent) * 1000))) & 0xFF

        if key in KEY_QUIT:
            break
        if key in KEY_FLAP and not calibrating:
            world.flap()
        elif key in KEY_RESET:
            world.reset()
            effects.clear()
        elif key in KEY_CALIBRATE and vision is not None:
            cal = cal or Calibrator(conf, sens)
            cal.restart()
        elif key in KEY_DIFFICULTY:
            world.set_difficulty(KEY_DIFFICULTY[key])
            effects.clear()
        elif key in KEY_AUTOPILOT:
            bot = None if bot is not None else Autopilot()
            if bot is not None and world.state is not GameState.PLAYING:
                world.reset()
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    if vision is not None:
        if args.record and vision.trace:
            args.record.write_text(json.dumps(vision.trace))
            print(f"wrote {len(vision.trace)} samples to {args.record}")
        vision.stop()
    cv2.destroyAllWindows()
    save_best(world.best)
    print(f"best score: {world.best}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("no cameras found (on macOS, check Privacy & Security > Camera)")
            return 1
        for c in cams:
            print(f"  [{c.index}] {c.width}x{c.height} @ {c.fps:.0f}fps")
        return 0
    try:
        return run(args)
    except CameraError as exc:
        print(f"\ncamera error: {exc}")
        print("\n  ...or play without the camera:  flappy --keyboard")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
