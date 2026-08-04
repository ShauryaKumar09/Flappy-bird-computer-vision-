# Flappy Vision

Flappy Bird you play by flapping your arms at a webcam. MediaPipe Pose tracks
your shoulders and wrists; a downward arm sweep flaps the bird.

```
┌──────────────────┬────────────────────────────────┐
│   GAME           │   CAMERA + arm skeleton        │
│   512x720        │   768x576                      │
│                  ├────────────────────────────────┤
│                  │   HUD: flap meter, fps, latency│
└──────────────────┴────────────────────────────────┘
```

## Run it

```bash
uv run python scripts/fetch_assets.py   # once: pose model + sprites
uv run flappy                           # calibrate, then flap
```

| | |
|---|---|
| `uv run flappy` | camera control, with calibration |
| `uv run flappy -s high` | **flaps not registering? start here** |
| `uv run flappy -d chill` | gentlest preset (0.76 flaps/sec) |
| `uv run flappy --keyboard` | no camera; spacebar to flap |
| `uv run flappy -d chill\|normal\|hard\|classic` | starting difficulty |
| `uv run flappy --list-cameras` | pick an index if 0 is your iPhone |
| `uv run flappy --mode raise` | fall back to static arm-up detection |
| `uv run flappy --record trace.json` | dump a landmark trace for offline tuning |

Keys: `1`-`4` difficulty · `SPACE` flap · `C` recalibrate · `R` restart · `A` autopilot · `Q` quit

## If it's hard to play

Two independent knobs, and they fix different problems:

- **Flaps get missed** → `-s high`. That's detector sensitivity.
- **Too many flaps needed / tiring** → press `1` for chill. That's game physics.

Press **1-4 in game** to switch difficulty — no restart, no CLI flag needed.
The picker is on the start and game-over screens.

| key | preset | gap | hover rate | a point every |
|---|---|---|---|---|
| `1` | `chill` | 460px | 0.76/s | 2.5s |
| `2` | `normal` *(default)* | 400px | 0.91/s | 2.1s |
| `3` | `hard` | 280px | 1.63/s | 1.9s |
| `4` | `classic` | 200px | 2.02/s | 1.5s |

Scoring pace is `pipe_spacing / pipe_speed`. Wide gaps buy slack, and the right
thing to spend it on is tempo — a point every ~2s plays like a game, a point
every ~5s plays like waiting.

The playfield is 624px and the bird's collision box is 36px, so `normal` leaves
about 10x the bird's height of slack and costs under one flap per second to
hover. `hard` and `classic` are where the original game's cruelty lives.

Watch the HUD while playing. `swing` shows how big your current flap reads; if
it says `FLAP BIGGER`, your motion is under the `min_span` floor. The `up`/`down`
ticks show the live band — in adaptive mode they move with you.

| sensitivity | smallest reliable flap | note |
|---|---|---|
| `low` | 1.60 shoulder-widths | big committed swings only |
| `normal` | 0.55 (~22cm of wrist travel) | default |
| `high` | 0.55, degrades gracefully to 0.35 | may false-fire on very noisy tracking |

The presets differ almost only in `min_span` — how much total swing counts as a
flap. That is the lever; moving the trigger height instead does nothing (see
below).

## Why it's a dupe, not a port

Real Flappy Bird's constants assume a zero-latency tap. An arm flap costs ~100ms
of pipeline plus ~200ms of physically moving your arms, and classic pipe gaps
leave roughly 4 frames of reaction margin — you'd be dying to latency, not skill.
So the physics is retuned and every constant is exposed. `--difficulty classic`
is there if you want the original's cruelty anyway.

## How the detection works

```
landmarks → arm elevation H (in shoulder-widths)   scale-invariant
          → one-euro filter                        jitter out, lag low
          → velocity dH/dt
          → Schmitt trigger + refractory           one flap, one event
```

Four choices carry most of the weight:

- **H is measured in shoulder-widths**, so thresholds hold whether you stand 1m
  or 3m from the camera. Verified exact to 1e-16 across a 3.5x size range.
- **The trigger keys on downward velocity**, not on a static "wrist above
  shoulder" line. It fires on the downstroke, where a real wing generates lift.
- **Hysteresis plus a refractory window.** On 10s of jitter parked exactly on the
  threshold, this fires ≤1 time; a bare `H > 0` edge test fires **78** times.
- **The band is adaptive**, tracking a decaying envelope (τ=2s) of your recent
  motion rather than sitting at absolute heights.

That last one was the fix for "this is impossible to play". Any *fixed* band
assumes you keep your arms in the same place all game — but arms sag as you
tire, and people drift between big wing flaps and holding them out to flap the
forearms. Drift out of a fixed band and the trigger silently goes quiet.
Tracking recent min/max makes the trigger care about the *shape* of the motion,
not its height: verified identical counts at arm heights from −1.2 to +1.2.

Two subtleties, both found by measurement rather than intuition:

- **Envelope decay must be relative (a time constant), not a fixed rate.** A
  fixed shoulder-widths/sec release outruns a small flap, collapsing the
  envelope between strokes so gentle flapping stops registering entirely.
- **`min_span` is the lever, not the trigger height.** Lowering `up_frac` from
  0.62 to 0.42 does not improve small-flap detection at all; lowering `min_span`
  from 0.30 to 0.22 takes an amplitude-0.55 flap from 3/10 to 10/10. A "high"
  preset that lowered *both* actually scored worse (9/10) than one that moved
  `min_span` alone (10/10).
- **That tradeoff has a floor.** Still arms with noisy tracking produce an
  envelope up to ~0.12 wide (measured at σ=0.08), so it cannot go arbitrarily
  low without catching jitter. That's what `-s high` trades.

Calibration scales `min_span` to your body and refuses to apply anything if your
arms-up and arms-down poses look the same.

`--fixed-thresholds` reverts to absolute thresholds if you want them.

## Verify

```bash
uv run python scripts/verify_physics.py    # dt-independence, collision, playability
uv run python scripts/verify_detector.py   # flap counting, chatter, calibration
uv run python scripts/verify_pipeline.py   # live camera soak: fps and latency
uv run python scripts/shot.py              # render stills to /tmp
uv run python scripts/replay.py trace.json # re-run a recorded trace
```

Measured on an M-series MacBook Pro: render loop **52.8 fps** while the vision
thread runs at **30.1 fps** with **8.0 ms** inference. Physics is dt-independent
to **0.00px** from 20 to 120 fps.

Record/replay exists so detector thresholds can be tuned against identical real
motion instead of re-flapping for every code change.

## Notes

- Python 3.14 works — MediaPipe 1.0.0 ships `py3-none` wheels.
- On macOS the first run prompts *your terminal* for camera access. If it opens
  but sees nothing, that's the permission, not the hardware.
- Sprites: [samuelcust/flappy-bird-assets](https://github.com/samuelcust/flappy-bird-assets).
  Inspired by [2bato/big-flappy-bird](https://github.com/2bato/big-flappy-bird).
