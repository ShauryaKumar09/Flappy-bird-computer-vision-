"""Death feedback: screen shake and a feather burst.

Cosmetic, but it is what makes a collision *read* as a collision on video rather
than the bird just stopping.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import config as cfg

GRAVITY = 900.0
SHAKE_DURATION = 0.35
SHAKE_AMPLITUDE = 14.0


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: int

    @property
    def alpha(self) -> float:
        return max(0.0, self.life / self.max_life)


@dataclass
class Effects:
    particles: list[Particle] = field(default_factory=list)
    shake_t: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    def burst(self, x: float, y: float, n: int = 22) -> None:
        for _ in range(n):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(60, 320)
            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed - 120,
                    life=(life := self.rng.uniform(0.4, 1.0)),
                    max_life=life,
                    size=self.rng.randint(2, 5),
                )
            )
        self.shake_t = SHAKE_DURATION

    def update(self, dt: float) -> None:
        self.shake_t = max(0.0, self.shake_t - dt)
        for p in self.particles:
            p.vy += GRAVITY * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0 and p.y < cfg.GAME_H]

    def shake_offset(self) -> tuple[int, int]:
        if self.shake_t <= 0:
            return (0, 0)
        # Decay to zero so the pane settles rather than snapping back.
        decay = self.shake_t / SHAKE_DURATION
        amp = SHAKE_AMPLITUDE * decay * decay
        return (
            int(self.rng.uniform(-amp, amp)),
            int(self.rng.uniform(-amp, amp)),
        )

    def clear(self) -> None:
        self.particles.clear()
        self.shake_t = 0.0
