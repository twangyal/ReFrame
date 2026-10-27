#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/reframe-uv-cache}"
export UV_LINK_MODE=copy
command -v uv >/dev/null
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python python3 .venv
fi
uv pip sync --python .venv/bin/python --require-hashes requirements-cpu.lock
.venv/bin/python -m unittest discover -s tests -v
