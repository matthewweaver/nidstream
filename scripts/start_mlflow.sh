#!/usr/bin/env bash
# Moved — use the unified start script instead:
#   bash scripts/start.sh          # local (MLflow + API + Streamlit via docker-compose)
#   bash scripts/start.sh --aws    # AWS (docker-compose + EMR tunnel)
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")"  && pwd)/start.sh" "$@"
