#!/usr/bin/env python3
"""Emit web/status.json: the single source of truth the dashboard renders.

Everything here is read from the real filesystem / nvidia-smi. Nothing is hand-written,
so the dashboard can never show progress that did not happen.
"""
from __future__ import annotations
import json, os, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/hf/track_1"
WEB = ROOT / "web"
STATE = ROOT / "state"
DEADLINE = datetime(2026, 11, 4, 17, 0, tzinfo=timezone(-__import__("datetime").timedelta(hours=5)))  # 17:00 EST

# Pipeline stages, in order. Each stage declares how to count its own completion.
STAGES = [
    ("env",        "环境与依赖",        lambda: 1 if (ROOT / "work/models/mhr_model.pt").exists() else 0, 1),
    ("data",       "数据下载与核对",     lambda: len(list((DATA / "videos/chunk-000/observation.images.exo_camera").glob("*.mp4"))), 30),
    ("mhr_gate",   "MHR 接口回环",       lambda: _flag("mhr_gate"), 1),
    ("frames",     "抽帧 / 评分窗口缓存", lambda: _count("frames"), 30),
    ("seg",        "人+物分割",          lambda: _count("seg"), 30),
    ("kp2d",       "2D 关键点 (VitPose)", lambda: _count("kp2d"), 30),
    ("body",       "SAM 3D Body → MHR",  lambda: _count("body"), 30),
    ("objmesh",    "物体网格 (10 物体)",  lambda: _count("objmesh"), 10),
    ("objpose",    "物体 6D 跟踪",        lambda: _count("objpose"), 30),
    ("joint",      "联合优化 + 接触/穿透", lambda: _count("joint"), 30),
    ("pack",       "打包 parquet",        lambda: _count("pack"), 30),
    ("submit",     "Kaggle 提交",         lambda: _nsub(), 1),
]


def _flag(name: str) -> int:
    return 1 if (STATE / f"{name}.done").exists() else 0


def _count(stage: str) -> int:
    d = STATE / stage
    return len(list(d.glob("*.done"))) if d.is_dir() else 0


def _nsub() -> int:
    f = ROOT / "notes/submissions.md"
    if not f.exists():
        return 0
    return sum(1 for ln in f.read_text().splitlines() if ln.startswith("| 20"))


def gpus() -> list[dict]:
    try:
        q = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return []
    rows = []
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 6:
            continue
        rows.append(dict(index=int(p[0]), name=p[1].replace("NVIDIA ", ""), mem_used=int(p[2]),
                         mem_total=int(p[3]), util=int(p[4]), temp=int(p[5])))
    return rows


def gpu_procs() -> list[dict]:
    """Who else is on the GPU, and on WHICH card. The box is shared; this is not decoration."""
    try:
        uuid_rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=15).stdout.strip()
        by_uuid = {u.strip(): int(i) for i, u in (ln.split(",") for ln in uuid_rows.splitlines())}
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return []
    procs = []
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) < 4:
            continue
        try:
            owner = subprocess.run(["ps", "-o", "user=", "-p", p[1]], capture_output=True, text=True,
                                   timeout=5).stdout.strip() or "?"
        except Exception:
            owner = "?"
        procs.append(dict(gpu=by_uuid.get(p[0], -1), pid=int(p[1]), name=Path(p[2]).name,
                          mem=int(p[3]), user=owner, mine=owner == "tianyuan"))
    return procs


def objects() -> list[dict]:
    """10 objects x 3 episodes. The 3 episodes of one object SHARE one reconstructed mesh,
    so mesh progress is per object, not per episode."""
    f = DATA / "meta/episodes_metadata.jsonl"
    if not f.exists():
        return []
    rows: dict[str, dict] = {}
    for ln in f.open():
        r = json.loads(ln)
        o = rows.setdefault(r["object"], dict(name=r["object"], prompt=r["object_prompt"], episodes=[]))
        o["episodes"].append(r["episode_index"])
    for o in rows.values():
        o["mesh_done"] = (STATE / "objmesh" / f"{o['name']}.done").exists()
    return [rows[k] for k in sorted(rows)]


def failures() -> list[dict]:
    """A crashed episode must not look like one that simply has not started."""
    out = []
    for f in sorted(STATE.glob("*/*.fail")):
        out.append(dict(stage=f.parent.name, unit=f.stem,
                        message=f.read_text()[:400].strip(),
                        at=datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")))
    return out


def submissions() -> list[dict]:
    """The leaderboard is our only validation set (private_fraction=0), so the score history
    belongs on the dashboard, not only in a markdown table."""
    f = ROOT / "notes/submissions.md"
    if not f.exists():
        return []
    out = []
    for ln in f.read_text().splitlines():
        if not ln.startswith("| 20"):
            continue
        c = [x.strip() for x in ln.strip("|").split("|")]
        if len(c) < 8:
            continue
        num = lambda v: None if v in ("", "-", "pending") else float(v)
        try:
            out.append(dict(ts=c[0], commit=c[1], config=c[2], cd_h=num(c[3]), cd_o=num(c[4]),
                            acc_h=num(c[5]), acc_o=num(c[6]), pen=num(c[7])))
        except ValueError:
            continue
    return out


def blockers() -> list[dict]:
    """Parsed from the plan's blocker table, so there is exactly one place to edit them."""
    f = ROOT / "notes/00-plan-v2.md"
    if not f.exists():
        return []
    out = []
    for ln in f.read_text().splitlines():
        m = __import__("re").match(r"\| (B-\d) \| (.+?) \| (.+?) \|", ln)
        if m:
            status = m.group(3)
            out.append(dict(id=m.group(1), what=m.group(2).replace("**", ""),
                            status=status.replace("**", ""),
                            open="待办" in status or "未" in status))
    return out


def disk_breakdown() -> dict:
    """At 87% full, knowing which directory is the problem is what makes the alarm actionable."""
    out = {}
    for name in ("data", "work", "state", "web", "logs"):
        d = ROOT / name
        if not d.is_dir():
            continue
        try:
            out[name] = round(int(subprocess.run(["du", "-sb", str(d)], capture_output=True,
                                                 text=True, timeout=120).stdout.split()[0]) / 1e9, 2)
        except Exception:
            pass
    return out


def running() -> list[dict]:
    """Jobs in flight, with the log to look at. Silence and success look identical otherwise."""
    try:
        out = subprocess.run(["pgrep", "-af", "python3 src/"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:
        return []
    jobs = []
    for ln in out.splitlines():
        pid, _, cmd = ln.partition(" ")
        m = __import__("re").search(r"src/(\w+)\.py", cmd)
        if not m or m.group(1) == "status" or int(pid) == os.getpid():
            continue  # the status job is not progress
        log = ROOT / f"logs/{m.group(1)}.log"
        jobs.append(dict(pid=int(pid), script=m.group(1), cmd=cmd[:160],
                         log=str(log.relative_to(ROOT)) if log.exists() else None))
    return jobs


def leaderboard() -> dict:
    """Our live standing. Written by src/fetch_leaderboard.py, which needs the submit venv."""
    f = ROOT / "work/leaderboard.json"
    return json.loads(f.read_text()) if f.exists() else {}


def episodes() -> list[dict]:
    meta_f = DATA / "meta/episodes_metadata.jsonl"
    eps_f = DATA / "meta/episodes.jsonl"
    scored_f = ROOT / "work/scored_frames_summary.json"
    if not meta_f.exists():
        return []
    meta = {json.loads(l)["episode_index"]: json.loads(l) for l in meta_f.open()}
    lens = {json.loads(l)["episode_index"]: json.loads(l) for l in eps_f.open()} if eps_f.exists() else {}
    scored = json.load(scored_f.open()) if scored_f.exists() else {}
    stage_names = [s[0] for s in STAGES if s[0] in
                   ("frames", "seg", "kp2d", "body", "objpose", "joint", "pack")]
    out = []
    for i in sorted(meta):
        s = scored.get(str(i), {})
        done = [st for st in stage_names if (STATE / st / f"ep{i:06d}.done").exists()]
        failed = [st for st in stage_names if (STATE / st / f"ep{i:06d}.fail").exists()]
        out.append(dict(ep=i, object=meta[i]["object"], camera=meta[i]["camera"],
                        sequence=meta[i]["sequence_id"], prompt=meta[i]["object_prompt"],
                        task=(lens.get(i, {}).get("tasks") or [None])[0],
                        frames=lens.get(i, {}).get("length"),
                        scored=s.get("n"), stretches=s.get("stretches"),
                        first=s.get("first"), last=s.get("last"), spans=s.get("spans"),
                        thumb_frame=(s.get("first", 0) + s.get("last", 0)) // 2,
                        done=done, failed=failed))
    return out


def main() -> None:
    stages = []
    for key, label, fn, total in STAGES:
        try:
            n = int(fn())
        except Exception:
            n = 0
        d = STATE / key
        marks = sorted((f.stat().st_mtime for f in d.glob("*.done")) if d.is_dir() else [])
        rate = None
        if len(marks) >= 2 and marks[-1] > marks[0]:
            rate = round(3600.0 * (len(marks) - 1) / (marks[-1] - marks[0]), 1)  # units/hour
        stages.append(dict(key=key, label=label, done=min(n, total), total=total,
                           pct=round(100.0 * min(n, total) / total, 1),
                           last_at=datetime.fromtimestamp(marks[-1], timezone.utc).isoformat(timespec="seconds") if marks else None,
                           rate_per_hour=rate,
                           eta_hours=round((total - n) / rate, 1) if rate and n < total else None))
    du = shutil.disk_usage(ROOT)
    now = datetime.now(timezone.utc)
    log = ROOT / "notes/progress.md"
    status = dict(
        generated_at=now.isoformat(timespec="seconds"),
        deadline=DEADLINE.isoformat(),
        hours_left=round((DEADLINE - now).total_seconds() / 3600, 1),
        overall_pct=round(sum(s["pct"] for s in stages) / len(stages), 1),
        stages=stages,
        gpus=gpus(),
        gpu_procs=gpu_procs(),
        disk=dict(total_gb=round(du.total / 1e9), free_gb=round(du.free / 1e9),
                  used_pct=round(100 * du.used / du.total, 1), by_dir=disk_breakdown()),
        objects=objects(),
        blockers=blockers(),
        failures=failures(),
        submissions=submissions(),
        leaderboard=leaderboard(),
        running=running(),
        episodes=episodes(),
        artifacts=sorted(p.name for p in (WEB / "assets").glob("*")) if (WEB / "assets").is_dir() else [],
        log_tail=log.read_text().splitlines()[-40:] if log.exists() else [],
    )
    WEB.mkdir(exist_ok=True)
    (WEB / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))
    print(f"{now:%Y-%m-%d %H:%M:%S} overall {status['overall_pct']}%  "
          f"free {status['disk']['free_gb']}GB  gpus busy "
          f"{sum(1 for g in status['gpus'] if g['util'] > 5)}/{len(status['gpus'])}")


if __name__ == "__main__":
    main()
