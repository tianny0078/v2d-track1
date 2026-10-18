#!/bin/bash
cd /home/tianyuan/projects/v2d_track1
python3 src/acc_gate.py work/sub_v3 2>&1 | grep RESULT
python3 src/obj_hand_dist.py work/sub_track2 work/sub_v3 2>&1 | grep RESULT | awk '{print}' > logs/obj_hand_v3.txt; tail -n 3 logs/obj_hand_v3.txt
python3 - <<'PY'
import re
rows=[l for l in open('logs/obj_hand_v3.txt')]
import numpy as np
w2=[];w3=[];a2=[];a3=[]
for l in rows:
    m=re.findall(r"wrist\s+(\d+)/\s*(\d+) pelvis\s+(\d+) \|a\| ([\d.]+)", l)
    if len(m)==2: w2.append(int(m[0][0])); w3.append(int(m[1][0])); a2.append(float(m[0][3])); a3.append(float(m[1][3]))
print(f"RESULT wrist median over eps: track2 {np.median(w2):.0f} cm -> v3 {np.median(w3):.0f} cm; eps >80 cm: {sum(x>80 for x in w2)} -> {sum(x>80 for x in w3)}; |a| mean {np.mean(a2):.2f} -> {np.mean(a3):.2f}")
PY
python3 src/pen_gate.py work/sub_v3 cuda:1 2>&1 | grep RESULT
echo "=== DONE"
