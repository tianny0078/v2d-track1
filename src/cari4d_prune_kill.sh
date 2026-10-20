#!/bin/bash
# Kill the CARI4D pruner loop without a final pass (pattern lives here so callers never match it).
for p in $(pgrep -f "cari4d_prune.py --loop"); do kill "$p" 2>/dev/null; done
sleep 1
echo "pruner procs left: $(pgrep -fc 'cari4d_prune.py --loop')"
