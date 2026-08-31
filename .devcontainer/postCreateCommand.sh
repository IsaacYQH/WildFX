#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPOSITORY_DIR}"
uv pip install --python /home/u1/miniconda3/bin/python -r requirements.txt
timeout 10s python -c 'import reapy; print(f"Connected to REAPER project {reapy.Project().id}")'
