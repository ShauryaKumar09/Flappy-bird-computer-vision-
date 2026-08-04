"""Live soak test of the full pipeline, minus the window.

The claim being tested: because capture+inference live on their own thread, the
render loop stays fast regardless of camera rate. The reference implementation
this project improves on ran inference inline, so its 60fps loop actually ran at
camera speed minus inference time.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from flappy_vision import config as cfg
from flappy_vision.game import World
from flappy_vision.pose import VisionThread
from flappy_vision.render import Assets, render_camera, render_game, render_hud

DURATION = 12.0
TARGET_FPS = 60


def main() -> int:
    assets = Assets()
    world = World(diff=cfg.NORMAL)
    conf = cfg.DetectorConfig()

    vision = VisionThread(camera=0, conf=conf)
    vision.start()
    print(f"vision thread started; soaking for {DURATION:.0f}s...\n")

    frame_ms: list[float] = []
    latencies: list[float] = []
    flaps = 0
    poses = 0
    frames = 0

    t_end = time.perf_counter() + DURATION
    prev = time.perf_counter()
    render_fps = float(TARGET_FPS)

    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        dt = t0 - prev
        prev = t0
        render_fps = 0.9 * render_fps + 0.1 / max(dt, 1e-6)

        if vision.error is not None:
            raise vision.error

        snap = vision.latest()
        for ev in vision.drain_flaps():
            world.flap()
            flaps += 1
            latencies.append((time.perf_counter() - ev.t_capture) * 1000)

        world.update(dt)

        game = render_game(world, assets)
        if snap is not None:
            poses += bool(snap.have_pose)
            top = render_camera(snap, conf)
            hud = render_hud(
                snap,
                conf,
                {
                    "render_fps": render_fps,
                    "vision_fps": snap.vision_fps,
                    "infer_ms": snap.infer_ms,
                    "latency_ms": latencies[-1] if latencies else 0.0,
                    "flaps": flaps,
                },
            )
            _ = np.hstack((game, np.vstack((top, hud))))
        frames += 1
        frame_ms.append((time.perf_counter() - t0) * 1000)

        # Cooperative pacing, standing in for cv2.waitKey.
        slack = (1 / TARGET_FPS) - (time.perf_counter() - t0)
        if slack > 0:
            time.sleep(slack)

    snap = vision.latest()
    vision.stop()

    render_hz = frames / DURATION
    print(f"render loop      : {render_hz:6.1f} fps over {frames} frames")
    print(f"  frame cost     : mean {statistics.mean(frame_ms):5.2f} ms  "
          f"p95 {np.percentile(frame_ms, 95):5.2f} ms  max {max(frame_ms):5.2f} ms")
    if snap is not None:
        print(f"vision thread    : {snap.vision_fps:6.1f} fps  "
              f"inference {snap.infer_ms:.1f} ms")
    print(f"frames with pose : {poses}/{frames}")
    print(f"flaps detected   : {flaps}")
    if latencies:
        print(f"  flap latency   : mean {statistics.mean(latencies):5.1f} ms  "
              f"p95 {np.percentile(latencies, 95):5.1f} ms")

    checks = {
        "render loop >= 50 fps": render_hz >= 50,
        "frame cost p95 < 16ms": float(np.percentile(frame_ms, 95)) < 16.0,
    }
    if latencies:
        checks["flap latency < 150ms"] = statistics.mean(latencies) < 150
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not latencies:
        print("  [SKIP] flap latency - nobody flapped at the camera")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
