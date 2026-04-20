#!/usr/bin/env bash
# Archon CLI launcher for Linux / macOS
# Usage:
#   chmod +x archon.sh
#   sudo cp archon.sh /usr/local/bin/archon   # or symlink
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/archon/cli.py" "$@"
