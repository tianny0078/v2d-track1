#!/bin/bash
# Kill running push-out jobs (pattern lives here so the caller's shell never matches it).
for p in $(pgrep -f "src/cari4d_pushout.py"); do kill "$p" 2>/dev/null; done
sleep 2
echo "pushout procs left: $(pgrep -fc 'src/cari4d_pushout.py')"
