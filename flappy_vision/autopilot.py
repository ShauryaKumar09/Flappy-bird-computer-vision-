"""A competent scripted player.

Used two ways: as a gameplay sanity check in scripts/verify_physics.py, and as an
attract-mode demo so the game plays itself when nobody is in front of the camera.
"""

from __future__ import annotations

from . import config as cfg
from .game import GameState, World


class Autopilot:
    """Lead controller: flap when the bird is *predicted* to fall below the gap.

    Reacting to current position alone is too late - the bird carries a lot of
    downward velocity. Two guards keep it from killing itself: it won't stack
    flaps while already climbing, and it won't fire if the flap would carry it
    through the top pipe.
    """

    def __init__(self, lookahead: float = 0.18, cooldown: float = 0.15) -> None:
        self.lookahead = lookahead
        self.cooldown = cooldown
        self.t = 0.0
        self._last_flap = -1.0

    def step(self, world: World, dt: float) -> bool:
        """Advance internal clock and return True if it wants to flap this tick."""
        self.t += dt
        d = world.diff
        bird = world.bird

        # Off the menu / game-over screen the bird is frozen, so the predictive
        # rule below can never fire and attract mode would sit there forever.
        # Nudge it to start, rate-limited so a restart isn't instant.
        if world.state is not GameState.PLAYING:
            if self.t - self._last_flap >= max(self.cooldown, 0.8):
                self._last_flap = self.t
                return True
            return False

        # Keep targeting a pipe until the bird's box has fully cleared it.
        bird_left = cfg.BIRD_X - cfg.BIRD_W / 2 + 6
        ahead = [p for p in world.pipes if p.x + cfg.PIPE_W > bird_left]
        if ahead:
            target = ahead[0].gap_y
            ceiling = ahead[0].gap_y - d.pipe_gap / 2
        else:
            target, ceiling = cfg.PLAY_H / 2, 0.0

        rise = d.flap_impulse**2 / (2 * d.gravity)
        half_h = cfg.BIRD_H / 2 - 6

        predicted = bird.y + bird.vy * self.lookahead
        headroom_ok = (bird.y - rise - half_h) > ceiling + 8
        not_climbing = bird.vy > -150
        off_cooldown = self.t - self._last_flap >= self.cooldown

        if predicted > target and headroom_ok and not_climbing and off_cooldown:
            self._last_flap = self.t
            return True
        return False
