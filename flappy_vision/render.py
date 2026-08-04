"""Rendering. Everything composites into plain numpy BGR arrays - no pygame.

Sprites are pre-split into a BGR array plus a float alpha mask so the blit is a
single vectorised lerp rather than a per-pixel loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config as cfg
from .game import GameState, World

FONT = cv2.FONT_HERSHEY_DUPLEX


@dataclass
class Sprite:
    bgr: np.ndarray  # (h, w, 3) uint8
    alpha: np.ndarray  # (h, w) float32 in [0, 1]

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.bgr.shape[:2]
        return w, h

    @classmethod
    def load(cls, path: Path, size: tuple[int, int] | None = None) -> "Sprite":
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"missing sprite {path} - run scripts/fetch_assets.py")
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        if img.shape[2] == 4:
            return cls(np.ascontiguousarray(img[:, :, :3]), img[:, :, 3].astype(np.float32) / 255.0)
        return cls(img, np.ones(img.shape[:2], np.float32))

    def flipped_v(self) -> "Sprite":
        return Sprite(cv2.flip(self.bgr, 0), cv2.flip(self.alpha, 0))

    def rotated(self, degrees: float) -> "Sprite":
        """Rotate about centre, expanding the canvas so nothing is clipped."""
        h, w = self.bgr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        bgr = cv2.warpAffine(self.bgr, m, (nw, nh), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(self.alpha, m, (nw, nh), flags=cv2.INTER_LINEAR)
        return Sprite(bgr, alpha)


def blit(dst: np.ndarray, sprite: Sprite, x: float, y: float) -> None:
    """Alpha-composite `sprite` onto `dst` at (x, y), clipped to bounds."""
    x, y = int(round(x)), int(round(y))
    h, w = sprite.bgr.shape[:2]
    dh, dw = dst.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x + w), min(dh, y + h)
    if x0 >= x1 or y0 >= y1:
        return

    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)

    a = sprite.alpha[sy0:sy1, sx0:sx1, None]
    roi = dst[y0:y1, x0:x1]
    np.copyto(roi, (sprite.bgr[sy0:sy1, sx0:sx1] * a + roi * (1.0 - a)).astype(np.uint8))


def blit_centered(dst: np.ndarray, sprite: Sprite, cx: float, cy: float) -> None:
    w, h = sprite.size
    blit(dst, sprite, cx - w / 2, cy - h / 2)


class Assets:
    def __init__(self, d: Path = cfg.ASSETS_DIR) -> None:
        self.bird = [
            Sprite.load(d / f"yellowbird-{f}.png", (cfg.BIRD_W, cfg.BIRD_H))
            for f in ("downflap", "midflap", "upflap")
        ]
        pipe = Sprite.load(d / "pipe-green.png", (cfg.PIPE_W, cfg.PIPE_SPRITE_H))
        self.pipe_bottom = pipe
        self.pipe_top = pipe.flipped_v()
        self.background = Sprite.load(d / "background-day.png", (cfg.GAME_W, cfg.GAME_H))
        # Tile the ground at its native 3:1 aspect instead of stretching one copy
        # across the pane, which visibly distorts the pixel art.
        tile = Sprite.load(d / "base.png", (cfg.GROUND_TILE_W, cfg.GROUND_H))
        n = cfg.GAME_W // cfg.GROUND_TILE_W + 2
        self.ground = Sprite(
            np.hstack([tile.bgr] * n), np.hstack([tile.alpha] * n)
        )
        self.digits = [Sprite.load(d / f"{i}.png", (36, 54)) for i in range(10)]
        self.gameover = Sprite.load(d / "gameover.png", (384, 84))
        self.message = Sprite.load(d / "message.png", (258, 374))


def draw_number(dst: np.ndarray, assets: Assets, value: int, cx: float, y: float) -> None:
    digits = [assets.digits[int(c)] for c in str(value)]
    total = sum(s.size[0] for s in digits)
    x = cx - total / 2
    for s in digits:
        blit(dst, s, x, y)
        x += s.size[0]


def render_game(
    world: World,
    assets: Assets,
    effects=None,
    chrome: bool = True,
) -> np.ndarray:
    """Draw the game pane. `chrome=False` omits the menu/game-over overlays, so
    the calibration screen can own the pane without the 'Get Ready' art behind it."""
    shake = effects.shake_offset() if effects is not None else (0, 0)
    frame = np.empty((cfg.GAME_H, cfg.GAME_W, 3), np.uint8)
    blit(frame, assets.background, *shake)

    gap = world.diff.pipe_gap
    for p in world.pipes:
        top_bottom = p.gap_y - gap / 2
        blit(frame, assets.pipe_top, p.x + shake[0], top_bottom - cfg.PIPE_SPRITE_H + shake[1])
        blit(frame, assets.pipe_bottom, p.x + shake[0], p.gap_y + gap / 2 + shake[1])

    # Scroll within one tile; the strip is wide enough to cover the pane at any offset.
    off = world.ground_scroll % cfg.GROUND_TILE_W
    blit(frame, assets.ground, -off + shake[0], cfg.PLAY_H + shake[1])

    bird_sprite = assets.bird[world.bird.frame_index].rotated(world.bird.rotation)
    blit_centered(frame, bird_sprite, cfg.BIRD_X + shake[0], world.bird.y + shake[1])

    if effects is not None:
        # Feathers, so warm yellow like the bird. Fade by shrinking rather than
        # darkening - darkening toward black reads as soot against a bright sky.
        for p in effects.particles:
            r = max(1, int(round(p.size * p.alpha)))
            cv2.circle(
                frame,
                (int(p.x + shake[0]), int(p.y + shake[1])),
                r,
                (60, 220, 255),
                -1,
                cv2.LINE_AA,
            )

    if not chrome:
        return frame

    if world.state is GameState.PLAYING:
        draw_number(frame, assets, world.score, cfg.GAME_W / 2, 60)
    elif world.state is GameState.MENU:
        blit_centered(frame, assets.message, cfg.GAME_W / 2, cfg.GAME_H / 2 - 90)
        draw_difficulty_picker(frame, world, cfg.GAME_H * 0.72)
    else:
        _draw_game_over(frame, assets, world)

    return frame


def draw_difficulty_picker(dst: np.ndarray, world: World, y: float) -> None:
    """Row of selectable presets, shown whenever the game is not in play."""
    presets = cfg.DIFFICULTY_ORDER
    pad, h = 8, 30
    boxes = []
    for i, d in enumerate(presets):
        label = f"{i + 1} {d.name.upper()}"
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        boxes.append((label, tw + 18))

    total = sum(w for _, w in boxes) + pad * (len(boxes) - 1)
    x = cfg.GAME_W / 2 - total / 2
    y0 = int(y)

    for (label, w), d in zip(boxes, presets):
        selected = d is world.diff
        bg = cfg.C_ACCENT if selected else (52, 48, 44)
        fg = cfg.C_BLACK if selected else cfg.C_DIM
        cv2.rectangle(dst, (int(x), y0), (int(x + w), y0 + h), bg, -1)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.putText(
            dst,
            label,
            (int(x + (w - tw) / 2), y0 + (h + th) // 2),
            FONT,
            0.45,
            fg,
            1,
            cv2.LINE_AA,
        )
        x += w + pad

    _label(dst, "press 1-4 to change difficulty", cfg.GAME_W / 2, y0 + h + 24, 0.4)


def _draw_game_over(dst: np.ndarray, assets: Assets, world: World) -> None:
    # Without a scrim the text lands on top of pipes and is unreadable.
    cv2.addWeighted(dst, 0.45, np.zeros_like(dst), 0.0, 0.0, dst)

    cx = cfg.GAME_W / 2
    blit_centered(dst, assets.gameover, cx, cfg.GAME_H * 0.26)

    _label(dst, "SCORE", cx, cfg.GAME_H * 0.26 + 80, 0.55)
    draw_number(dst, assets, world.score, cx, cfg.GAME_H * 0.26 + 96)

    _label(dst, "BEST", cx, cfg.GAME_H * 0.26 + 200, 0.55)
    draw_number(dst, assets, world.best, cx, cfg.GAME_H * 0.26 + 216)

    _label(dst, "FLAP TO RETRY", cx, cfg.GAME_H * 0.26 + 300, 0.7)
    draw_difficulty_picker(dst, world, cfg.GAME_H * 0.26 + 330)


def _label(dst: np.ndarray, text: str, cx: float, y: float, scale: float = 0.6) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, 1)
    org = (int(cx - tw / 2), int(y))
    cv2.putText(dst, text, org, FONT, scale, cfg.C_BLACK, 3, cv2.LINE_AA)
    cv2.putText(dst, text, org, FONT, scale, cfg.C_WHITE, 1, cv2.LINE_AA)


def render_calibration(pane: np.ndarray, cal) -> np.ndarray:
    """Overlay the calibration prompt on top of the game pane."""
    cv2.addWeighted(pane, 0.30, np.zeros_like(pane), 0.0, 0.0, pane)
    cx, cy = cfg.GAME_W / 2, cfg.GAME_H / 2

    title, hint = cal.prompt()
    _label(pane, "CALIBRATION", cx, cy - 110, 0.55)
    _label(pane, title, cx, cy - 40, 1.3)
    _label(pane, hint, cx, cy + 4, 0.6)

    # Progress bar fills only while actually sampling.
    bw, bh = 300, 12
    x0, y0 = int(cx - bw / 2), int(cy + 40)
    cv2.rectangle(pane, (x0, y0), (x0 + bw, y0 + bh), (60, 55, 50), -1)
    cv2.rectangle(pane, (x0, y0), (x0 + int(bw * cal.progress), y0 + bh), cfg.C_ACCENT, -1)

    if cal.failed:
        _label(pane, cal.failed, cx, cy + 90, 0.5)
    _label(pane, "press C to redo", cx, cfg.PLAY_H - 28, 0.45)
    return pane


# ------------------------------------------------------------ camera pane ----

_STATE_COLOR = {
    "DOWN": cfg.C_DIM,
    "ARMED": cfg.C_WARN,
    "RECOVER": cfg.C_GOOD,
}


def render_camera(snap, conf: cfg.DetectorConfig) -> np.ndarray:
    """Webcam view with the arm skeleton drawn on top."""
    from .pose import ARM_BONES  # local import keeps render.py importable headless

    pane = cv2.resize(snap.frame, (cfg.CAM_W, cfg.CAM_H), interpolation=cv2.INTER_LINEAR)

    if not snap.points:
        cv2.addWeighted(pane, 0.35, np.zeros_like(pane), 0.0, 0.0, pane)
        _label(pane, "STEP INTO FRAME", cfg.CAM_W / 2, cfg.CAM_H / 2, 0.9)
        return pane

    px = {i: (int(x * cfg.CAM_W), int(y * cfg.CAM_H)) for i, (x, y) in snap.points.items()}

    for a, b in ARM_BONES:
        if a in px and b in px:
            cv2.line(pane, px[a], px[b], cfg.C_BLACK, 6, cv2.LINE_AA)
            cv2.line(pane, px[a], px[b], cfg.C_WHITE, 2, cv2.LINE_AA)

    colour = _STATE_COLOR.get(snap.state.name, cfg.C_DIM)
    for i, p in px.items():
        cv2.circle(pane, p, 8, cfg.C_BLACK, -1, cv2.LINE_AA)
        cv2.circle(pane, p, 6, colour, -1, cv2.LINE_AA)

    # The shoulder line is the reference the elevation is measured against, so
    # showing it makes the whole metric legible at a glance.
    from .pose import L_SHOULDER, R_SHOULDER

    if L_SHOULDER in px and R_SHOULDER in px:
        y = (px[L_SHOULDER][1] + px[R_SHOULDER][1]) // 2
        cv2.line(pane, (0, y), (cfg.CAM_W, y), cfg.C_ACCENT, 1, cv2.LINE_AA)

    return pane


# -------------------------------------------------------------------- HUD ----

# Spans arms-hanging-down (about -1.3) through fully overhead (about +1.8).
H_MIN, H_MAX = -1.6, 2.2


def _meter_x(h: float, x0: int, w: int) -> int:
    frac = (h - H_MIN) / (H_MAX - H_MIN)
    return x0 + int(np.clip(frac, 0.0, 1.0) * w)


def render_hud(snap, conf: cfg.DetectorConfig, stats: dict) -> np.ndarray:
    pane = np.full((cfg.HUD_H, cfg.CAM_W, 3), cfg.C_PANEL, np.uint8)

    x0, w, y = 24, 420, 56
    cv2.putText(pane, "ARM ELEVATION", (x0, 30), FONT, 0.45, cfg.C_DIM, 1, cv2.LINE_AA)

    cv2.rectangle(pane, (x0, y), (x0 + w, y + 26), (46, 42, 38), -1)
    if snap is not None and snap.have_pose:
        fill = _meter_x(snap.h, x0, w)
        cv2.rectangle(pane, (x0, y), (fill, y + 26), _STATE_COLOR.get(snap.state.name), -1)

    # Ticks come from the snapshot, not the config: in adaptive mode the band
    # tracks your recent motion, so drawing the static config values would lie.
    up = snap.h_up if snap is not None else conf.h_up
    down = snap.h_down if snap is not None else conf.h_down
    for h, tag, col in ((down, "down", cfg.C_BAD), (up, "up", cfg.C_GOOD)):
        tx = _meter_x(h, x0, w)
        cv2.line(pane, (tx, y - 6), (tx, y + 32), col, 2, cv2.LINE_AA)
        cv2.putText(pane, tag, (tx - 10, y - 12), FONT, 0.35, col, 1, cv2.LINE_AA)

    state = snap.state.name if snap is not None else "-"
    cv2.putText(
        pane, state, (x0, y + 62), FONT, 0.6, _STATE_COLOR.get(state, cfg.C_DIM), 1, cv2.LINE_AA
    )
    if snap is not None:
        cv2.putText(
            pane,
            f"H {snap.h:+5.2f}   dH/dt {snap.v:+6.2f}/s   "
            f"swing {snap.span:.2f}" + ("" if snap.active else "  - FLAP BIGGER"),
            (x0 + 90, y + 62),
            FONT,
            0.45,
            cfg.C_DIM,
            1,
            cv2.LINE_AA,
        )

    rx = x0 + w + 40
    rows = [
        ("render", f"{stats.get('render_fps', 0):5.1f} fps"),
        ("vision", f"{stats.get('vision_fps', 0):5.1f} fps"),
        ("inference", f"{stats.get('infer_ms', 0):5.1f} ms"),
        ("latency", f"{stats.get('latency_ms', 0):5.1f} ms"),
        ("flaps", f"{stats.get('flaps', 0):5d}"),
    ]
    for i, (k, v) in enumerate(rows):
        yy = 28 + i * 22
        cv2.putText(pane, k, (rx, yy), FONT, 0.42, cfg.C_DIM, 1, cv2.LINE_AA)
        cv2.putText(pane, v, (rx + 96, yy), FONT, 0.42, cfg.C_ACCENT, 1, cv2.LINE_AA)

    return pane
