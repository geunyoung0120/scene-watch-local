#!/bin/zsh
cd "$(dirname "$0")" || exit 1
.venv/bin/python scripts/manage_server.py stop
