"""Generate private deployment settings and an independent evaluation key."""

import argparse
import os
from pathlib import Path
import secrets


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--public", action="store_true")
parser.add_argument("--clef-port", type=int, default=8082)
parser.add_argument("--nox-port", type=int, default=8083)
args = parser.parse_args()
root = args.root.resolve(strict=True)
env = root / "source" / ".env"
if env.exists():
    raise SystemExit("Existing private settings preserved; initialization not repeated")
os.umask(0o077)
env.write_text(
    f"EVAL_ROOT={root}\n"
    f"DECISION_EVAL_API_KEY={secrets.token_urlsafe(48)}\n"
    f"LISTEN_ADDRESS={'0.0.0.0' if args.public else '127.0.0.1'}\n"
    f"CLEF_PORT={args.clef_port}\nNOX_PORT={args.nox_port}\n"
    "MAX_INPUT_TOKENS=2048\nCLEF_GPU_FRACTION=0.55\nNOX_GPU_FRACTION=0.30\n"
)
print("Private settings and independent evaluation API key initialized")
