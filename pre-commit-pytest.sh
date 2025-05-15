#!/bin/bash
echo "Files passed to pytest:" "$@"
source ./.venv/bin/activate
exec pytest "$@"
