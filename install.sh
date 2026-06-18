#!/usr/bin/env bash
set -e
REPO="https://github.com/rsastore/rsa-agentic.git"
DIR="$HOME/neural"
if [ -d "$DIR" ]; then echo "Already installed at $DIR"; exit 0; fi
command -v git >/dev/null 2>&1 || { echo "Need git"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Need python3"; exit 1; }
git clone --depth=1 "$REPO" "$DIR"
pip3 install -q prompt_toolkit rich requests 2>/dev/null || pip install -q prompt_toolkit rich requests
echo "Installed to $DIR"
echo "Run: $DIR/neural.py"
