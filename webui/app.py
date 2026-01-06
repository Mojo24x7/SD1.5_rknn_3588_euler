#!/usr/bin/env python3
import os, sys, re, json, time, hashlib, threading, subprocess
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

from flask import Flask, request, jsonify, send_file, render_template, abort

ROOT = Path(__file__).resolve().parents[1]
RKIMG = ROOT / "rkimg.py"
DATA_DIR = ROOT / "webui" / "data"
THUMB_DIR = Path("/tmp/rkimg_thumbs"); THUMB_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png",".jpg",".jpeg",".webp",".bmp"}
BLOCKED_PREFIX = ("/proc", "/sys", "/dev")

app = Flask(
    __name__,
    template_folder=str(ROOT/"webui/templates"),
    static_folder=str(ROOT/"webui/static")
)

JOBS = {}
JOBS_LOCK = threading.Lock()

STEP_PAT = re.compile(r"(?:step|inference step)\s*[:#]?\s*(\d+)\s*(?:/|of)\s*(\d+)", re.I)

# How many jobs to keep in memory (completed). Keeps RAM stable.
KEEP_DONE_JOBS = 40

def safe_abspath(p: str) -> str:
    p = os.path.abspath(os.path.expanduser(str(p)))
    for pref in BLOCKED_PREFIX:
        if p == pref or p.startswith(pref + "/"):
            raise ValueError("Blocked path")
    return p

def run_cmd(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def rkimg_help(subcmd: str):
    cmd = [sys.executable, str(RKIMG), subcmd, "--help"]
    _, out = run_cmd(cmd, cwd=str(ROOT))
    return out

def parse_help(help_text: str):
    usage_block = ""
    m = re.search(r"usage:\s+.*?(?:\n\n|$)", help_text, re.S|re.I)
    if m:
        usage_block = m.group(0)

    bracket_chunks = re.findall(r"\[[^\]]+\]", usage_block)
    optional_flags = set(re.findall(r"--[a-z0-9\-]+", " ".join(bracket_chunks)))
    all_flags = set(re.findall(r"--[a-z0-9\-]+", usage_block))
    required_flags = set([f for f in all_flags if f not in optional_flags and f != "--help"])

    opts = []
    for ln in help_text.splitlines():
        if not re.match(r"^\s{2,}--", ln):
            continue
        parts = re.split(r"\s{2,}", ln.strip(), maxsplit=2)
        head = parts[0]
        head_tokens = head.split()
        flag = head_tokens[0]
        if flag == "--help":
            continue

        dest = flag[2:].replace("-", "_")
        meta = " ".join(head_tokens[1:]) if len(head_tokens) > 1 else ""
        help_part = parts[-1] if len(parts) >= 2 else ""

        choices = None
        cm = re.search(r"\{([^}]+)\}", meta)
        if cm:
            choices = [c.strip() for c in cm.group(1).split(",") if c.strip()]

        default = None
        dm = re.search(r"\(default:\s*([^)]+)\)", help_part)
        if dm:
            default = dm.group(1).strip()

        typ = "text"
        if choices:
            typ = "choice"
        elif not meta:
            typ = "bool"
        else:
            if dest in ("steps","tile","overlap","scale","upscale","seed"):
                typ = "int"
            elif dest in ("guidance","strength"):
                typ = "float"

        opts.append({
            "flag": flag,
            "dest": dest,
            "meta": meta,
            "help": help_part,
            "choices": choices,
            "default": default,
            "required": (flag in required_flags),
            "type": typ,
        })
    return opts

def ensure_history():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hist = DATA_DIR / "history.jsonl"
    if not hist.exists():
        hist.write_text("")
    return hist

def list_dir(path: str, show_hidden: bool):
    path = safe_abspath(path)
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(path)

    items = []
    for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        name = child.name
        if not show_hidden and name.startswith("."):
            continue
        ext = child.suffix.lower()
        is_img = child.is_file() and ext in IMAGE_EXTS
        st = child.stat()
        items.append({
            "name": name,
            "path": str(child),
            "type": "dir" if child.is_dir() else "file",
            "size": st.st_size if child.is_file() else None,
            "mtime": st.st_mtime,
            "is_image": is_img,
            "thumb_url": ("/thumb?path=" + quote(str(child), safe="")) if is_img else None
        })
    return {"path": str(p), "items": items}

def thumb_for(path: str, max_px=256):
    path = safe_abspath(path)
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(path)
    if p.suffix.lower() not in IMAGE_EXTS:
        raise ValueError("Not an image")

    key = hashlib.sha1(f"{p}:{p.stat().st_mtime}:{max_px}".encode()).hexdigest()
    out = THUMB_DIR / f"{key}.jpg"
    if out.exists():
        return out

    from PIL import Image
    with Image.open(p) as im:
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        im.save(out, "JPEG", quality=85)
    return out

def serve_image_file(path: str):
    path = safe_abspath(path)
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(path)
    if p.suffix.lower() not in IMAGE_EXTS:
        raise ValueError("Not an image")
    ext = p.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg",".jpeg") else "image/webp" if ext == ".webp" else "application/octet-stream"
    return send_file(str(p), mimetype=mime)

def build_rkimg_command(cmd: str, args: dict):
    if cmd not in ("generate","edit","upscale","facefix"):
        raise ValueError("Invalid cmd")
    out = [sys.executable, str(RKIMG), cmd]
    for k, v in args.items():
        if v is None or v == "":
            continue
        flag = "--" + k.replace("_","-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
        else:
            out += [flag, str(v)]
    return out

def newest_image_in_dir(out_dir: str, after_ts: float):
    try:
        d = Path(safe_abspath(out_dir))
        if not d.exists() or not d.is_dir():
            return None
        best = None
        best_m = 0.0
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            m = p.stat().st_mtime
            if m >= after_ts and m >= best_m:
                best_m = m
                best = str(p)
        return best
    except Exception:
        return None

def prune_done_jobs_locked():
    # keep only last KEEP_DONE_JOBS completed jobs (by created timestamp)
    done = [j for j in JOBS.values() if j.get("state") == "done"]
    if len(done) <= KEEP_DONE_JOBS:
        return
    done_sorted = sorted(done, key=lambda x: x.get("created", 0.0), reverse=True)
    keep_ids = set([x["job_id"] for x in done_sorted[:KEEP_DONE_JOBS]])
    for jid in list(JOBS.keys()):
        if JOBS[jid].get("state") == "done" and jid not in keep_ids:
            JOBS.pop(jid, None)

def history_find(job_id: str):
    """
    Read history.jsonl from disk and return record matching job_id.
    This lets clients reopen results after board restart.
    """
    hist = ensure_history()
    if not hist.exists():
        return None
    try:
        # read from end for speed (small enough anyway)
        lines = hist.read_text().splitlines()
        for ln in reversed(lines[-800:]):  # cap search
            try:
                r = json.loads(ln)
            except:
                continue
            if r.get("job_id") == job_id:
                return r
    except:
        pass
    return None

def job_worker(job_id: str, cmd_list, expected_out=None, out_dir=None, src_in=None, total_steps_hint=None):
    started = time.time()
    proc = subprocess.Popen(cmd_list, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    with JOBS_LOCK:
        JOBS[job_id]["state"] = "running"
        JOBS[job_id]["started"] = started
        JOBS[job_id]["total_steps"] = total_steps_hint
        JOBS[job_id]["expected_out"] = expected_out
        JOBS[job_id]["out_dir"] = out_dir
        JOBS[job_id]["src_in"] = src_in

    seen_step = 0
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            with JOBS_LOCK:
                JOBS[job_id]["log"].append(line)
                # cap memory
                if len(JOBS[job_id]["log"]) > 6000:
                    JOBS[job_id]["log"] = JOBS[job_id]["log"][-2500:]

            m = STEP_PAT.search(line)
            if m:
                s = int(m.group(1)); t = int(m.group(2))
                seen_step = max(seen_step, s)
                with JOBS_LOCK:
                    JOBS[job_id]["steps_done"] = seen_step
                    JOBS[job_id]["total_steps"] = t

        rc = proc.wait()
    except Exception as ex:
        rc = 1
        with JOBS_LOCK:
            JOBS[job_id]["log"].append(f"\n[webui] error: {ex}\n")

    elapsed = time.time() - started

    out_path = None
    out_ok = False

    if expected_out:
        try:
            cand = safe_abspath(expected_out)
            if Path(cand).exists():
                out_path = cand
                out_ok = True
        except Exception:
            pass

    if (not out_ok) and out_dir:
        cand = newest_image_in_dir(out_dir, started - 0.5)
        if cand:
            out_path = cand
            out_ok = True

    with JOBS_LOCK:
        JOBS[job_id]["state"] = "done"
        JOBS[job_id]["rc"] = rc
        JOBS[job_id]["elapsed_s"] = round(elapsed, 3)
        JOBS[job_id]["out_exists"] = out_ok
        JOBS[job_id]["out_path"] = out_path if out_ok else None
        prune_done_jobs_locked()

    # history persist
    hist = ensure_history()
    with JOBS_LOCK:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "job_id": job_id,
            "cmd": cmd_list,
            "rc": rc,
            "elapsed_s": round(elapsed,3),
            "src_in": src_in,
            "expected_out": expected_out,
            "out_dir": out_dir,
            "out_path": out_path if out_ok else None,
            "log_tail": "".join(JOBS[job_id]["log"][-250:])
        }
    with hist.open("a") as f:
        f.write(json.dumps(rec) + "\n")

@app.get("/")
def index():
    return render_template("index.html", base=str(ROOT))

@app.get("/api/schema")
def api_schema():
    schema = {
        "generate": parse_help(rkimg_help("generate")),
        "edit": parse_help(rkimg_help("edit")),
        "upscale": parse_help(rkimg_help("upscale")),
        "facefix": parse_help(rkimg_help("facefix")),
    }
    return jsonify(schema)

@app.get("/api/list")
def api_list():
    path = request.args.get("path", str(ROOT))
    show_hidden = request.args.get("hidden","0") == "1"
    try:
        return jsonify(list_dir(path, show_hidden))
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400

@app.get("/thumb")
def api_thumb():
    path = request.args.get("path","")
    if not path:
        abort(400)
    try:
        t = thumb_for(path)
        return send_file(str(t), mimetype="image/jpeg")
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400

@app.get("/file")
def api_file():
    path = request.args.get("path","")
    if not path:
        abort(400)
    try:
        return serve_image_file(path)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 400


@app.post("/api/mask/save")
def api_mask_save():
    payload = request.get_json(force=True) or {}
    data_url = payload.get("png_data_url", "")
    if not data_url.startswith("data:image/png;base64,"):
        return jsonify({"error": "png_data_url must be a data:image/png;base64,... URL"}), 400

    b64 = data_url.split(",", 1)[1].strip()
    try:
        import base64, io
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        return jsonify({"error": "Invalid base64"}), 400

    masks_dir = DATA_DIR / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path = masks_dir / f"mask_{int(time.time()*1000)}.png"

    try:
        # Normalize to a real binary L mask (0/255), robust to RGBA/RGB/etc.
        from PIL import Image
        import numpy as np
        import io

        im = Image.open(io.BytesIO(raw))

        if im.mode == "RGBA":
            arr = np.array(im)
            a = arr[..., 3]
            # If alpha actually contains paint info, use alpha; otherwise use luminance
            if a.min() != a.max():
                g = a
            else:
                g = np.array(im.convert("L"))
        else:
            g = np.array(im.convert("L"))

        bw = (g > 127).astype(np.uint8) * 255
        Image.fromarray(bw, mode="L").save(out_path)

    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

    return jsonify({"path": str(out_path)})


@app.post("/api/mask/invert")
def api_mask_invert():
    payload = request.get_json(force=True) or {}
    path = payload.get("path", "")
    if not path:
        return jsonify({"error": "path is required"}), 400

    try:
        p = Path(safe_abspath(path))
        if not p.exists() or not p.is_file():
            return jsonify({"error": "file not found"}), 400
        if p.suffix.lower() not in IMAGE_EXTS:
            return jsonify({"error": "mask must be an image"}), 400

        from PIL import Image, ImageOps
        import numpy as np

        im = Image.open(p).convert("L")
        inv = ImageOps.invert(im)

        # keep it binary after invert
        arr = np.array(inv)
        bw = (arr > 127).astype(np.uint8) * 255

        masks_dir = DATA_DIR / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        out_path = masks_dir / f"mask_invert_{int(time.time()*1000)}.png"
        Image.fromarray(bw, mode="L").save(out_path)

        return jsonify({"path": str(out_path)})

    except Exception as ex:
        return jsonify({"error": str(ex)}), 400

@app.post("/api/run")
def api_run():
    payload = request.get_json(force=True)
    cmd = payload.get("cmd")
    args = payload.get("args", {}) or {}

    expected_out = args.get("out")
    out_dir = None
    if expected_out:
        try:
            out_dir = str(Path(expected_out).expanduser().resolve().parent)
        except Exception:
            out_dir = None

    src_in = None
    if cmd == "edit":
        src_in = args.get("init")
    elif cmd in ("upscale","facefix"):
        src_in = args.get("input")

    total_steps_hint = None
    if cmd in ("generate","edit"):
        if "steps" in args and str(args["steps"]).strip():
            try:
                total_steps_hint = int(args["steps"])
            except:
                pass

    cmd_list = build_rkimg_command(cmd, args)
    job_id = hashlib.sha1(f"{time.time()}:{cmd_list}".encode()).hexdigest()[:10]

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "state": "queued",
            "cmd": cmd_list,
            "created": time.time(),
            "started": None,
            "rc": None,
            "elapsed_s": None,
            "log": [],
            "steps_done": 0,
            "total_steps": total_steps_hint,
            "expected_out": expected_out,
            "out_dir": out_dir,
            "src_in": src_in,
            "out_exists": False,
            "out_path": None,
        }

    t = threading.Thread(
        target=job_worker,
        args=(job_id, cmd_list, expected_out, out_dir, src_in, total_steps_hint),
        daemon=True
    )
    t.start()

    return jsonify({"ok": True, "job_id": job_id, "cmd": cmd_list})

@app.get("/api/jobs/active")
def api_jobs_active():
    with JOBS_LOCK:
        # queued/running jobs
        active = []
        for j in JOBS.values():
            if j.get("state") in ("queued", "running"):
                active.append({
                    "job_id": j.get("job_id"),
                    "state": j.get("state"),
                    "created": j.get("created"),
                    "started": j.get("started"),
                    "steps_done": j.get("steps_done", 0),
                    "total_steps": j.get("total_steps"),
                    "expected_out": j.get("expected_out"),
                    "out_dir": j.get("out_dir"),
                    "src_in": j.get("src_in"),
                    "cmd": j.get("cmd"),
                })
        active = sorted(active, key=lambda x: x.get("created", 0.0), reverse=True)
        return jsonify(active)

@app.post("/api/job/<job_id>/forget")
def api_job_forget(job_id):
    # optional endpoint if you ever add "forget job" button
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS.pop(job_id, None)
    return jsonify({"ok": True})

@app.get("/api/job/<job_id>")
def api_job(job_id):
    with JOBS_LOCK:
        j = JOBS.get(job_id)

    if not j:
        # fallback to history (board restart recovery)
        rec = history_find(job_id)
        if not rec:
            return jsonify({"error":"not found"}), 404
        # return a compatible "done" response
        return jsonify({
            "job_id": rec.get("job_id"),
            "state": "done",
            "rc": rec.get("rc"),
            "elapsed_s": rec.get("elapsed_s"),
            "cmd": rec.get("cmd"),
            "steps_done": None,
            "total_steps": None,
            "pct": None,
            "log_tail": rec.get("log_tail",""),
            "src_in": rec.get("src_in"),
            "expected_out": rec.get("expected_out"),
            "out_dir": rec.get("out_dir"),
            "out_exists": bool(rec.get("out_path")),
            "out_path": rec.get("out_path"),
        })

    log_tail = "".join(j["log"][-500:])
    total = j["total_steps"]
    done = j["steps_done"]
    pct = None
    if total and total > 0:
        pct = max(0, min(100, int((done/total)*100)))

    return jsonify({
        "job_id": job_id,
        "state": j["state"],
        "rc": j["rc"],
        "elapsed_s": j["elapsed_s"],
        "cmd": j["cmd"],
        "steps_done": done,
        "total_steps": total,
        "pct": pct,
        "log_tail": log_tail,
        "src_in": j.get("src_in"),
        "expected_out": j.get("expected_out"),
        "out_dir": j.get("out_dir"),
        "out_exists": j.get("out_exists"),
        "out_path": j.get("out_path"),
    })

@app.get("/api/history")
def api_history():
    hist = ensure_history()
    lines = hist.read_text().splitlines()
    last = lines[-50:] if len(lines) > 50 else lines
    recs = []
    for ln in reversed(last):
        try:
            recs.append(json.loads(ln))
        except:
            pass
    return jsonify(recs)

if __name__ == "__main__":
    host = os.environ.get("RKIMG_UI_HOST","0.0.0.0")
    port = int(os.environ.get("RKIMG_UI_PORT","7860"))
    app.run(host=host, port=port, debug=False)
