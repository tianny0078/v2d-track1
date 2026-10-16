#!/usr/bin/env bash
# Refresh status.json and the offline snapshot embedded in index.html.
# Publishing is done by the Claude session (it holds the artifact URL); this script
# only prepares the files, so it is safe to run on a timer regardless.
set -euo pipefail
cd /home/tianyuan/projects/v2d_track1
work/venv-submit/bin/python src/fetch_leaderboard.py || echo "leaderboard fetch failed (keeping previous snapshot)"
python3 src/status.py
python3 - <<'PY'
import re, json
s = open('web/index.html', encoding='utf-8').read()
inj = json.dumps(json.load(open('web/status.json', encoding='utf-8')),
                 ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
s2, n = re.subn(r'(<script type="application/json" id="snapshot">).*?(</script>)',
                lambda m: m.group(1) + inj + m.group(2), s, count=1, flags=re.S)
if n != 1:
    raise SystemExit('snapshot block not found in index.html -- did the page change?')
open('web/index.html', 'w', encoding='utf-8').write(s2)
print('snapshot refreshed')
PY
