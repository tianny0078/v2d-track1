#!/usr/bin/env bash
# disk_cleanup.sh -- periodic, idempotent, SAFE disk cleanup for v2d_track1 (user tianyuan).
#
#   usage:  src/disk_cleanup.sh [--dry-run] [--current-session DIR]
#
# What it does (each step is a no-op when there is nothing to do):
#   1. pip cache purge
#   2. ~/.cache/torch_extensions/* build dirs older than 7 days
#   3. /tmp/claude-*/ scratch session dirs + bash-edit-diff dirs older than 3 days, EXCEPT the current session
#   4. huggingface hub *.incomplete leftovers (failed downloads) older than 1 h
#   5. /tmp/tmp*.obj leaked by export_episodes.py/object_track.py (budget_mesh temp meshes),
#      owned by us, older than 1 h, not currently open by any process
#   6. __pycache__ dirs inside the project (not data/, not work/venv-submit)
#   7. logs/ files larger than 50 MB -> gzip (skipped if a process has the file open)
#   8. obsolete export dirs work/sub_* that are NOT in KEEP_SUBS and NOT named in notes/submissions.md
#      (only the directories; *.parquet and *.kaggle-receipt.json files are NEVER touched), plus work/stale_ep1
#   9. conda clean --all (only if conda exists and no conda/pip install is running)
#
# It never touches: data/, notes/, state/, work/models, work/kit, work/frames, HF model snapshots,
# anything under /home/tianyuan/projects/v2d (Track-3 agent) or /mnt/nas3, or any file not owned by $USER.
set -uo pipefail

# ---------------------------------------------------------------- config
PROJ=/home/tianyuan/projects/v2d_track1
WORK=$PROJ/work
# export dirs that are submitted or submission candidates -- edit this list as the project moves on
KEEP_SUBS=(sub_place7 sub_track2 sub_v4b sub_moge2 sub_track1 sub_v5 sub_v6 sub_v7)
CURRENT_SESSION=${CLAUDE_SESSION_DIR:-/tmp/claude-1017/-home-tianyuan/97442652-26ed-4696-bcac-ae17101ef4fa}
SCRATCH_MAX_DAYS=3          # /tmp/claude-* scratch older than this is removed
TORCH_EXT_MAX_DAYS=7
TMP_OBJ_MIN_AGE_MIN=60      # leaked /tmp/tmp*.obj must be at least this old
LOG_GZ_MB=50
HF_HUB=$HOME/.cache/huggingface/hub

DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-n) DRY=1 ;;
    --current-session) CURRENT_SESSION=$2; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- helpers
freed_kb=0
say()  { printf '%s\n' "$*"; }
kb()   { du -sk -- "$@" 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
gb()   { awk -v k="$1" 'BEGIN{printf "%.2f", k/1024/1024}'; }
# rmpath <path>: remove a file/dir (dry-run aware) and account its size
rmpath() {
  local p=$1 s
  [[ -e $p || -L $p ]] || return 0
  s=$(kb "$p"); freed_kb=$((freed_kb + s))
  if (( DRY )); then say "  [dry] rm -rf $p ($(gb "$s") GB)"; else rm -rf -- "$p" && say "  rm -rf $p ($(gb "$s") GB)"; fi
}
# open_files: every path currently open by any process of ours (for the /tmp and log checks)
OPEN_LIST=$(lsof -u "$USER" -nP -Fn 2>/dev/null | sed -n 's/^n//p' | sort -u)
is_open() { grep -qxF -- "$1" <<<"$OPEN_LIST"; }

say "== disk_cleanup $(date '+%F %T')  dry-run=$DRY  user=$USER"
say "== df before"; df -h / | tail -1
avail_before=$(df -k / | awk 'NR==2{print $4}')

# ---------------------------------------------------------------- 1. pip cache
say "-- 1. pip cache"
if command -v pip >/dev/null; then
  s=$(kb "$HOME/.cache/pip"); freed_kb=$((freed_kb + s))
  if (( DRY )); then say "  [dry] pip cache purge ($(gb "$s") GB)"; else pip cache purge >/dev/null 2>&1; say "  pip cache purge ($(gb "$s") GB)"; fi
fi

# ---------------------------------------------------------------- 2. torch_extensions
say "-- 2. torch_extensions build dirs > ${TORCH_EXT_MAX_DAYS}d"
if [[ -d $HOME/.cache/torch_extensions ]]; then
  while IFS= read -r -d '' d; do rmpath "$d"; done \
    < <(find "$HOME/.cache/torch_extensions" -mindepth 2 -maxdepth 2 -type d -mtime +"$TORCH_EXT_MAX_DAYS" -print0 2>/dev/null)
fi

# ---------------------------------------------------------------- 3. claude scratch
say "-- 3. /tmp/claude-* scratch > ${SCRATCH_MAX_DAYS}d (keeping $CURRENT_SESSION)"
for base in /tmp/claude-*; do
  [[ -d $base && -O $base ]] || continue
  # session dirs: /tmp/claude-<uid>/<-cwd-slug>/<session-uuid>
  while IFS= read -r -d '' d; do
    [[ $d == "$CURRENT_SESSION" || $CURRENT_SESSION == "$d"/* ]] && continue
    rmpath "$d"
  done < <(find "$base" -mindepth 2 -maxdepth 2 -type d -path "$base/-*" -mtime +"$SCRATCH_MAX_DAYS" -print0 2>/dev/null)
  # bash-edit-diff dirs
  [[ -d $base/bash-edit-diff ]] && while IFS= read -r -d '' d; do rmpath "$d"; done \
    < <(find "$base/bash-edit-diff" -mindepth 1 -maxdepth 1 -type d -mtime +"$SCRATCH_MAX_DAYS" -print0 2>/dev/null)
done

# ---------------------------------------------------------------- 4. HF .incomplete
say "-- 4. huggingface *.incomplete"
[[ -d $HF_HUB ]] && while IFS= read -r -d '' f; do rmpath "$f"; done \
  < <(find "$HF_HUB" -name '*.incomplete' -type f -mmin +60 -print0 2>/dev/null)

# ---------------------------------------------------------------- 5. leaked /tmp/tmp*.obj
say "-- 5. leaked /tmp/tmp*.obj (ours, > ${TMP_OBJ_MIN_AGE_MIN} min, not open)"
n=0; s=0
while IFS= read -r -d '' f; do
  is_open "$f" && { say "  skip (open): $f"; continue; }
  k=$(kb "$f"); s=$((s + k)); n=$((n + 1))
  (( DRY )) || rm -f -- "$f"
done < <(find /tmp -maxdepth 1 -type f -user "$USER" -name 'tmp*.obj' -mmin +"$TMP_OBJ_MIN_AGE_MIN" -print0 2>/dev/null)
freed_kb=$((freed_kb + s)); say "  $([[ $DRY == 1 ]] && echo '[dry] ')$n files, $(gb "$s") GB"

# ---------------------------------------------------------------- 6. __pycache__
say "-- 6. __pycache__ in project"
n=0; s=0
while IFS= read -r -d '' d; do
  k=$(kb "$d"); s=$((s + k)); n=$((n + 1)); (( DRY )) || rm -rf -- "$d"
done < <(find "$PROJ" -type d -name __pycache__ -not -path "$PROJ/data/*" -not -path "$WORK/venv-submit/*" -prune -print0 2>/dev/null)
freed_kb=$((freed_kb + s)); say "  $([[ $DRY == 1 ]] && echo '[dry] ')$n dirs, $(gb "$s") GB"

# ---------------------------------------------------------------- 7. big logs
say "-- 7. logs > ${LOG_GZ_MB} MB -> gzip"
while IFS= read -r -d '' f; do
  is_open "$f" && { say "  skip (open): $f"; continue; }
  k=$(kb "$f")
  if (( DRY )); then say "  [dry] gzip $f ($(gb "$k") GB)"; else gzip -9 -- "$f" && say "  gzip $f ($(gb "$k") GB)"; fi
  freed_kb=$((freed_kb + k * 9 / 10))   # text logs compress ~10x
done < <(find "$PROJ/logs" -type f -size +"${LOG_GZ_MB}"M ! -name '*.gz' -print0 2>/dev/null)

# ---------------------------------------------------------------- 8. obsolete work/sub_* dirs
say "-- 8. obsolete work/sub_* dirs (KEEP: ${KEEP_SUBS[*]} + anything named in notes/submissions.md)"
for d in "$WORK"/sub_*/; do
  d=${d%/}; [[ -d $d ]] || continue
  name=$(basename "$d"); keep=0
  for k in "${KEEP_SUBS[@]}"; do [[ $name == "$k" ]] && keep=1; done
  grep -qw -- "$name" "$PROJ/notes/submissions.md" 2>/dev/null && keep=1
  if (( keep )); then say "  keep $name"; else rmpath "$d"; fi
done
rmpath "$WORK/stale_ep1"

# ---------------------------------------------------------------- 9. conda
say "-- 9. conda clean"
if command -v conda >/dev/null; then
  if pgrep -fa 'conda (install|update|create)|pip install' >/dev/null; then say "  skip: an install is running"
  elif (( DRY )); then say "  [dry] conda clean --all --yes"
  else conda clean --all --yes >/dev/null 2>&1 && say "  conda clean --all --yes"; fi
else say "  (no conda)"; fi

# ---------------------------------------------------------------- summary
say "== df after"; df -h / | tail -1
avail_after=$(df -k / | awk 'NR==2{print $4}')
if (( DRY )); then
  say "== would free about $(gb "$freed_kb") GB (dry-run; nothing changed)"
else
  say "== freed $(gb $((avail_after - avail_before))) GB (df delta); accounted $(gb "$freed_kb") GB"
fi
