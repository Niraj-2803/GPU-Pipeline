#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${CAMEX_PYTHON:-python3}"
exec "$PYTHON" -m camex_pipeline.cli "$@"
