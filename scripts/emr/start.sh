#!/usr/bin/env bash
# Moved — use the unified start script instead:
#   bash scripts/start.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/start.sh" "$@"
