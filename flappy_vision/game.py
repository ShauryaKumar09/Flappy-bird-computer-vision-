"""Game simulation: bird, pipes, collision, scoring.

Pure simulation - no rendering, no input handling, no OpenCV. Everything advances
by an explicit dt in seconds so the physics is framerate-independent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto

from . import config as cfg


class GameState(Enum):
    MENU = auto()
    PLAYING = auto()
    GAME_OVER = auto()


@dataclass
class Bird:
    y: float = cfg.PLAY_H / 2
    vy: float = 0.0
    anim_t: float = 0.0

    @property
    def frame_index(self) -> int:
        return int(self.anim_t * 10) % 3

    @property
    def rotation(self) -> float:
        """Nose up when rising, pitch down as it falls. Degrees, CCW positive."""
        return max(-70.0, min(25.0, -self.vy * 0.06))

    def flap(self, diff: cfg.Difficulty) -> None:
        self.vy = diff.flap_impulse

    def update(self, dt: float, diff: cfg.Difficulty) -> None:
        # Exact kinematic step for constant acceleration: y += v*dt + a*dt^2/2.
        # Plain semi-implicit Euler (y += v*dt after updating v) accumulates O(dt)
        # position error, which makes the arc visibly different at 20 vs 120 FPS.
        # This form reproduces the same parabola at any dt.
        v0 = self.vy
        v1 = min(v0 + diff.gravity * dt, cfg.TERMINAL_VELOCITY)
        if v1 < cfg.TERMINAL_VELOCITY:
            self.y += v0 * dt + 0.5 * diff.gravity * dt * dt
        else:
            # Clamped this step: integrate the average velocity instead.
            self.y += 0.5 * (v0 + v1) * dt
        self.vy = v1
        self.anim_t += dt

    def rect(self) -> tuple[float, float, float, float]:
        """Collision box (x, y, w, h), inset a little to be forgiving."""
        pad = 6
        return (
            cfg.BIRD_X - cfg.BIRD_W / 2 + pad,
            self.y - cfg.BIRD_H / 2 + pad,
            cfg.BIRD_W - 2 * pad,
            cfg.BIRD_H - 2 * pad,
        )


@dataclass
class Pipe:
    x: float
    gap_y: float
    passed: bool = False

    def rects(self, gap: int) -> tuple[tuple, tuple]:
        top_h = self.gap_y - gap / 2
        bottom_y = self.gap_y + gap / 2
        return (
            (self.x, 0.0, float(cfg.PIPE_W), top_h),
            (self.x, bottom_y, float(cfg.PIPE_W), cfg.PLAY_H - bottom_y),
        )


def _overlaps(a: tuple, b: tuple) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


@dataclass
class World:
    diff: cfg.Difficulty = cfg.NORMAL
    state: GameState = GameState.MENU
    bird: Bird = field(default_factory=Bird)
    pipes: list[Pipe] = field(default_factory=list)
    score: int = 0
    best: int = 0
    ground_scroll: float = 0.0
    since_spawn: float = 0.0
    rng: random.Random = field(default_factory=random.Random)
    # Set on the frame the bird dies, so the renderer can react.
    just_died: bool = False

    def reset(self) -> None:
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        # First pipe arrives after a beat, not immediately.
        self.since_spawn = -self.diff.pipe_spacing * 0.4
        self.state = GameState.PLAYING
        self.just_died = False

    def flap(self) -> None:
        """The single control input. Doubles as start/restart."""
        if self.state is GameState.PLAYING:
            self.bird.flap(self.diff)
        else:
            self.reset()
            self.bird.flap(self.diff)

    def _spawn(self) -> None:
        # Margin keeps a gap from hugging the ceiling or ground. With the large
        # forgiving gaps it has to stay small, or there is no vertical variety
        # left to place them in.
        margin = 40
        half = self.diff.pipe_gap / 2
        lo = int(half + margin)
        hi = int(cfg.PLAY_H - half - margin)
        if lo >= hi:  # gap nearly fills the playfield; centre it
            lo = hi = int(cfg.PLAY_H / 2)
        self.pipes.append(Pipe(x=float(cfg.GAME_W), gap_y=float(self.rng.randint(lo, hi))))

    def _collided(self) -> bool:
        if self.bird.y - cfg.BIRD_H / 2 < 0:
            return True
        if self.bird.y + cfg.BIRD_H / 2 > cfg.PLAY_H:
            return True
        br = self.bird.rect()
        return any(
            _overlaps(br, r) for p in self.pipes for r in p.rects(self.diff.pipe_gap)
        )

    def update(self, dt: float) -> None:
        dt = min(dt, cfg.MAX_DT)
        self.just_died = False

        if self.state is not GameState.PLAYING:
            # Idle animation keeps the menu/game-over screen alive.
            self.bird.anim_t += dt
            self.ground_scroll += self.diff.pipe_speed * dt * 0.35
            return

        self.bird.update(dt, self.diff)

        travel = self.diff.pipe_speed * dt
        self.ground_scroll += travel

        # Distance-based spawning: pipe spacing stays constant if speed changes.
        self.since_spawn += travel
        if self.since_spawn >= self.diff.pipe_spacing:
            self.since_spawn -= self.diff.pipe_spacing
            self._spawn()

        for p in self.pipes:
            p.x -= travel
            if not p.passed and p.x + cfg.PIPE_W < cfg.BIRD_X:
                p.passed = True
                self.score += 1

        self.pipes = [p for p in self.pipes if p.x > -cfg.PIPE_W]

        if self._collided():
            self.state = GameState.GAME_OVER
            self.just_died = True
            self.best = max(self.best, self.score)
