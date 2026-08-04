"""Download the pose model and Flappy Bird sprites. Idempotent - skips what exists."""

import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

# python.org framework builds don't trust the macOS system keychain, so point
# urllib at certifi's bundle explicitly (certifi ships as a mediapipe dep).
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

SPRITE_BASE = "https://raw.githubusercontent.com/samuelcust/flappy-bird-assets/master/sprites"
SPRITES = [
    "yellowbird-downflap.png",
    "yellowbird-midflap.png",
    "yellowbird-upflap.png",
    "pipe-green.png",
    "background-day.png",
    "background-night.png",
    "base.png",
    "gameover.png",
    "message.png",
    *[f"{i}.png" for i in range(10)],
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get  {dest.name}", flush=True)
    with urllib.request.urlopen(url, context=_SSL_CTX) as resp:
        dest.write_bytes(resp.read())


def main() -> int:
    print("pose model:")
    fetch(MODEL_URL, MODELS / "pose_landmarker_lite.task")
    print("sprites:")
    for name in SPRITES:
        fetch(f"{SPRITE_BASE}/{name}", ASSETS / name)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
