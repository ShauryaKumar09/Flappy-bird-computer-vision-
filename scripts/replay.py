"""Re-run a recorded landmark trace through the detector.

Record once (`flappy --record trace.json`), then tune thresholds against the same
real motion as many times as you like - no re-flapping, and results are
comparable between runs because the input is identical.

    uv run python scripts/replay.py trace.json
    uv run python scripts/replay.py trace.json --h-up 0.8 --v-thresh 2.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flappy_vision import config as cfg
from flappy_vision.pose import FlapDetector


def replay(samples: list[dict], conf: cfg.DetectorConfig) -> list[float]:
    det = FlapDetector(conf)
    t0 = samples[0]["t"]
    return [s["t"] - t0 for s in samples if det.update(s["h"], s["t"])]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", type=Path)
    p.add_argument("--h-up", type=float, default=None)
    p.add_argument("--h-down", type=float, default=None)
    p.add_argument("--v-thresh", type=float, default=None)
    p.add_argument("--mode", choices=("flap", "raise"), default=None)
    args = p.parse_args()

    samples = json.loads(args.trace.read_text())
    if not samples:
        print("empty trace")
        return 1

    conf = cfg.DetectorConfig()
    for attr, val in (
        ("h_up", args.h_up),
        ("h_down", args.h_down),
        ("v_thresh", args.v_thresh),
        ("mode", args.mode),
    ):
        if val is not None:
            setattr(conf, attr, val)

    span = samples[-1]["t"] - samples[0]["t"]
    hs = [s["h"] for s in samples]
    print(f"{len(samples)} samples over {span:.1f}s  ({len(samples) / span:.1f} Hz)")
    print(f"raw H range: {min(hs):+.2f} .. {max(hs):+.2f}")
    print(f"thresholds : up {conf.h_up:.2f}  down {conf.h_down:.2f}  v {conf.v_thresh:.2f}")

    times = replay(samples, conf)
    print(f"\nflaps: {len(times)}  ({len(times) / span:.2f}/s)")
    if len(times) > 1:
        gaps = [b - a for a, b in zip(times, times[1:])]
        print(f"  gaps: min {min(gaps):.3f}s  mean {sum(gaps) / len(gaps):.3f}s  max {max(gaps):.3f}s")
        if min(gaps) < conf.refractory_s:
            print(f"  WARNING: a gap is under the {conf.refractory_s}s refractory window")
    print("  " + ", ".join(f"{t:.2f}" for t in times[:20]) + (" ..." if len(times) > 20 else ""))

    # Determinism: the detector carries filter state, so a second pass over the
    # same input must produce byte-identical timings or something is leaking.
    again = replay(samples, conf)
    print(f"\ndeterministic: {times == again}")
    return 0 if times == again else 1


if __name__ == "__main__":
    raise SystemExit(main())
