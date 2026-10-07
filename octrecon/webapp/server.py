"""Local web UI for octrecon (standard library only: http.server + threads).

    python launch.py              # starts the server and opens the browser
    python -m octrecon web        # same

API (JSON):
  GET  /api/system                 GPU/CPU/RAM status + default config
  GET  /api/browse?path=...        list sub-folders (and .mat/.json/.yaml files)
  POST /api/inspect                {volume_folder} -> scan summary + auto parameters
  POST /api/estimate/dispersion    {volume_folder, device, initial?}
  POST /api/estimate/focus         {volume_folder, device, dispersion_quadratic_term?}
  POST /api/preview                {volume_folder, zi, device, dispersion_quadratic_term?, focus?}
  POST /api/run                    {config}  -> starts a reconstruction job
  GET  /api/job?since=N            job state, progress, new log lines
  POST /api/cancel
  POST /api/open                   {path} open a folder in the OS file manager
The server binds to 127.0.0.1 only (local use).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

STATIC = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- helpers
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, bytes):
        return base64.b64encode(o).decode()
    if isinstance(o, Path):
        return str(o)
    return o


def system_info():
    from ..backend import gpu_available
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(), "gpu_available": False, "gpu_name": None, "gpu_note": None}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_GB"] = round(vm.total / 1e9, 1)
        info["ram_available_GB"] = round(vm.available / 1e9, 1)
    except Exception:
        pass
    from ..backend import mps_status
    try:
        import cupy  # noqa: F401
        has_cupy = True
    except Exception:
        has_cupy = False
    if has_cupy and gpu_available():
        import cupy as cp
        p = cp.cuda.runtime.getDeviceProperties(0)
        info.update(gpu_available=True, gpu_kind="cuda",
                    gpu_name=p["name"].decode() if isinstance(p["name"], bytes) else p["name"],
                    vram_GB=round(p["totalGlobalMem"] / 1e9, 1))
    elif sys.platform == "darwin":
        ok, msg = mps_status()
        if ok:
            info.update(gpu_available=True, gpu_kind="mps", gpu_name="Apple GPU (Metal/MPS)",
                        vram_GB=info.get("ram_total_GB"))
        else:
            info["gpu_note"] = (f"Apple GPU not usable: {msg}. Re-run start_mac.command to install PyTorch "
                                "(needs Apple silicon or a Metal GPU and macOS 13+).")
    elif has_cupy:
        info["gpu_note"] = ("CUDA is installed but the GPU cannot be initialised (on laptops this typically happens "
                            "after suspend/resume: reboot, or run `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`).")
    else:
        info["gpu_note"] = ("CuPy is not installed (CPU-only environment). On a machine with an NVIDIA GPU, "
                            "re-run the launcher so it can add GPU support, see README.")
    from ..pipeline import ReconConfig
    info["defaults"] = asdict(ReconConfig(volume_folder=""))
    return info


def browse(path: str | None):
    if not path:
        path = str(Path.home())
    p = Path(path).expanduser()
    if not p.exists():
        p = p.parent if p.parent.exists() else Path.home()
    if p.is_file():
        p = p.parent
    entries = []
    try:
        for c in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if c.name.startswith("."):
                continue
            try:
                if c.is_dir():
                    entries.append({"name": c.name, "path": str(c), "type": "dir",
                                    "is_volume": (c / "ScanInfo.json").exists(),
                                    "has_volume": (c / "OCTVolume" / "ScanInfo.json").exists()})
                elif c.suffix.lower() in (".mat", ".json", ".yaml", ".yml"):
                    entries.append({"name": c.name, "path": str(c), "type": "file"})
            except OSError:
                continue
            if len(entries) > 2000:
                break
    except PermissionError:
        pass
    roots = []
    if os.name == "nt":
        import string
        roots = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
    else:
        roots = ["/"] + [str(x) for x in (Path("/media"), Path("/mnt"), Path("/Volumes")) if x.exists()]
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "entries": entries,
            "is_volume": (p / "ScanInfo.json").exists(), "home": str(Path.home()), "roots": roots}


def open_in_file_manager(path: str):
    p = str(Path(path))
    if sys.platform.startswith("win"):
        os.startfile(p)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])


# --------------------------------------------------------------------------- job runner
class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.state = "idle"          # idle | starting | running | done | error | cancelled
        self.logs: list[str] = []
        self.progress = {}
        self.summary = None
        self.error = None
        self.config = None
        self.started = None
        self.finished = None
        self.cancel = threading.Event()
        self.thread = None

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self.lock:
            self.logs.append(line)
        print(line, flush=True)
        if self.config:
            try:
                out = Path(self.config["output_root"]) / self.config["output_name"]
                out.mkdir(parents=True, exist_ok=True)
                with open(out / f"{self.config['output_name']}.log", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def start(self, cfg_dict: dict):
        if self.state in ("starting", "running"):
            raise RuntimeError("a reconstruction is already running")
        self.reset()
        self.config = cfg_dict
        self.state = "starting"
        self.started = time.time()
        self.thread = threading.Thread(target=self._run, args=(cfg_dict,), daemon=True)
        self.thread.start()

    def _progress(self, **kw):
        with self.lock:
            self.progress = kw

    def _run(self, cfg_dict):
        from ..pipeline import Cancelled, ReconConfig, Reconstructor
        try:
            cfg = ReconConfig.from_dict(cfg_dict)
            self.log(f"starting reconstruction of {cfg.volume_folder}")
            rec = Reconstructor(cfg, log=self.log, progress=self._progress, cancel=self.cancel)
            self.state = "running"
            self.summary = rec.run()
            self.state = "done"
            self.log(f"finished: {self.summary.get('reconstruction_seconds', 0) / 60:.1f} min "
                     f"-> {self.summary.get('output_dir')}")
        except Cancelled:
            self.state = "cancelled"
            self.log("cancelled by user")
        except Exception as e:  # noqa: BLE001
            self.state = "error"
            self.error = f"{type(e).__name__}: {e}"
            self.log("ERROR " + self.error)
            self.log(traceback.format_exc())
        finally:
            self.finished = time.time()
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def status(self, since: int = 0):
        with self.lock:
            return {"state": self.state, "progress": self.progress, "logs": self.logs[since:],
                    "n_logs": len(self.logs), "summary": self.summary, "error": self.error,
                    "config": self.config, "started": self.started, "finished": self.finished}


JOB = Job()


# --------------------------------------------------------------------------- estimation endpoints
def _device(d):
    from ..backend import get_xp
    return get_xp(d or "auto")[1]


def api_estimate_dispersion(body):
    from ..estimation.dispersion import estimate_dispersion
    from ..estimation import preview
    res = estimate_dispersion(body["volume_folder"], device=_device(body.get("device")),
                              initial=body.get("initial"), log=lambda m: None)
    try:
        res["figure_png"] = preview.dispersion_curve_png(res)
    except Exception:
        pass
    return res


def api_estimate_focus(body):
    from ..estimation.focus import detect_focus
    from ..estimation import preview
    res = detect_focus(body["volume_folder"], dispersion_quadratic_term=body.get("dispersion_quadratic_term"),
                       device=_device(body.get("device")), log=lambda m: None)
    try:
        res["figure_png"] = preview.focus_profile_png(res)
    except Exception:
        pass
    return res


# --------------------------------------------------------------------------- interactive tools
_TOOL_LOCK = threading.Lock()
_VI_CACHE: dict = {}
_DT_CACHE: dict = {}
_FT_CACHE: dict = {}


def _vi(volume):
    from ..estimation.interactive import VolumeInfo, resolve_volume
    key = str(resolve_volume(volume))
    with _TOOL_LOCK:
        if key not in _VI_CACHE:
            _VI_CACHE.clear(); _DT_CACHE.clear(); _FT_CACHE.clear()
            _VI_CACHE[key] = VolumeInfo(key)
        return _VI_CACHE[key]


def api_tiles(body):
    return _vi(body["volume_folder"]).summary()


def _dispersion_tool(body):
    from ..estimation.interactive import DispersionTool
    vi = _vi(body["volume_folder"])
    folder = body["folder"]
    if folder not in vi.by_folder:
        raise ValueError(f"unknown tile folder {folder}")
    frame = int(body.get("frame", 1))
    if not 1 <= frame <= vi.hdr.size_y:
        raise ValueError(f"B-scan must be 1..{vi.hdr.size_y}")
    key = (folder, frame, bool(body.get("legacy_axis", False)))
    with _TOOL_LOCK:
        dt = _DT_CACHE.get(key)
    if dt is None:
        dt = DispersionTool(vi, folder, frame - 1, legacy_axis=key[2])
        with _TOOL_LOCK:
            if len(_DT_CACHE) > 8:
                _DT_CACHE.clear()
            _DT_CACHE[key] = dt
    return vi, dt


def api_dispersion_render(body):
    """Demo_DispersionCorrectionManual: ln|scan| with caxis [-5 6] for one slider value."""
    from ..estimation.interactive import DISPERSION_CLIM, beta_to_slider, gray_png, slider_to_beta
    vi, dt = _dispersion_tool(body)
    if body.get("beta") is not None:
        beta = float(body["beta"]); slider = beta_to_slider(beta)
    else:
        slider = float(body.get("slider", 2.0)); beta = slider_to_beta(slider)
    lg = dt.ln_image(beta)
    t = vi.by_folder[dt.folder]
    return {"png": gray_png(lg, *DISPERSION_CLIM), "beta": beta, "slider": slider, "shape": list(lg.shape),
            "sharpness": dt.sharpness(lg), "folder": dt.folder, "frame": dt.frame0 + 1,
            "tile": {"xi": t.xi, "yi": t.yi, "zi": t.zi, "x_mm": t.x_center_mm, "y_mm": t.y_center_mm,
                     "z_mm": t.z_depth_mm}, "system_used": dt.system_used}


def _focus_tool(volume):
    from ..estimation.interactive import FocusTool
    vi = _vi(volume)
    with _TOOL_LOCK:
        ft = _FT_CACHE.get("ft")
        if ft is None:
            ft = _FT_CACHE["ft"] = FocusTool(vi)
    return ft


def api_focus_setup(body):
    ft = _focus_tool(body["volume_folder"])
    s = ft.setup()
    s["folders"] = {f"{d['zi']},{xi}": ft.folder_for(d["zi"], xi)
                    for d in s["depths"] for xi in range(len(s["x_centers_mm"]))}
    return s


def api_focus_bscan(body):
    """yOCTMeasureFocusDrift renderTileBScan: dB image (float32) + 5/99.8 % base range."""
    from ..estimation.interactive import matlab_prctile
    from ..params import resolve_dispersion
    ft = _focus_tool(body["volume_folder"])
    zi, xi = int(body["zi"]), int(body["xi"])
    folder = ft.folder_for(zi, xi)
    fallback = False
    if folder is None:                       # legacy: fall back to the central X tile
        folder, xi, fallback = ft.folder_for(zi, ft.xi0), ft.xi0, True
        if folder is None:
            raise ValueError(f"No tile found for zDepth index {zi}")
    beta = body.get("dispersion_quadratic_term")
    beta = float(beta) if beta not in (None, "", "auto") else resolve_dispersion(ft.vi.si)[0]
    frame = max(1, min(ft.n_frames, int(body.get("frame", ft.frame_center))))
    img = ft.bscan_db(folder, frame, beta).astype(np.float32)
    lo, hi = matlab_prctile(img, [5, 99.8])
    if hi <= lo:
        hi = lo + 1
    return {"data_f32": base64.b64encode(np.ascontiguousarray(img).tobytes()).decode(), "shape": list(img.shape),
            "base_lo": float(lo), "base_hi": float(hi), "folder": folder, "xi": xi, "frame": frame,
            "fallback": fallback, "dispersion": beta}


def api_focus_fit(body):
    from ..estimation.interactive import drift_readout
    ft = _focus_tool(body["volume_folder"])
    z, pix = body.get("z_stage_mm", []), body.get("focus_pix", [])
    out = {"pred_pix": None, "readout": f"Clicked points: {len(pix)}\n(need >= 2 for slope)"}
    if len(pix) == 0:
        out["readout"] = "Drift so far: need >= 2 clicked points"
        return out
    try:
        q = body.get("z_query_mm")
        if q is not None:
            p, _ = ft.fit(z, pix, [float(q)])
            out["pred_pix"] = float(p[0])
        if len(pix) >= 2:
            _, d = ft.fit(z, pix, [0.0])
            out["readout"] = drift_readout(d, len(pix))
            out["diagnostics"] = d
    except Exception:
        if len(pix) >= 2:
            out["readout"] = f"Clicked points: {len(pix)}\n(no usable clicks for slope yet)"
    return out


def api_focus_finish(body):
    from ..estimation.interactive import finish_focus
    ft = _focus_tool(body["volume_folder"])
    out_dir = body.get("out_dir") or str(ft.vi.volume)
    res = finish_focus(ft, body.get("measurements", []), Path(out_dir), bool(body.get("also_volume", False)))
    return res


def api_preview(body):
    from ..estimation import preview
    r = preview.preview_bscan(body["volume_folder"], int(body.get("zi", 0)),
                              dispersion_quadratic_term=body.get("dispersion_quadratic_term"),
                              device=_device(body.get("device")))
    img = np.asarray(r.pop("image_db"), dtype=np.float64)
    # pixel-exact rendering (1 image row = 1 z pixel) so browser clicks map to z pixels
    fin = img[np.isfinite(img)]
    lo, hi = (np.percentile(fin, [5, 99.8]) if fin.size else (0.0, 1.0))
    g = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = _io.BytesIO()
    plt.imsave(buf, g, cmap="gray", vmin=0, vmax=1, format="png")
    r["png"] = buf.getvalue()
    r["shape"] = list(img.shape)
    r.setdefault("z0", 0)
    return r


# --------------------------------------------------------------------------- HTTP handler
class Handler(BaseHTTPRequestHandler):
    server_version = "octrecon"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(_jsonable(payload)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/static/"):
                f = (STATIC / u.path[len("/static/"):]).resolve()
                if STATIC.resolve() in f.parents and f.exists():
                    ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
                    return self._send(200, f.read_bytes(), ctype)
                return self._send(404, {"error": "not found"})
            if u.path == "/api/system":
                return self._send(200, system_info())
            if u.path == "/api/browse":
                return self._send(200, browse(q.get("path", [None])[0]))
            if u.path == "/api/job":
                return self._send(200, JOB.status(int(q.get("since", ["0"])[0])))
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/inspect":
                from ..params import inspect_volume
                return self._send(200, inspect_volume(body["volume_folder"]))
            if u.path == "/api/estimate/dispersion":
                return self._send(200, api_estimate_dispersion(body))
            if u.path == "/api/estimate/focus":
                return self._send(200, api_estimate_focus(body))
            if u.path == "/api/tiles":
                return self._send(200, api_tiles(body))
            if u.path == "/api/dispersion/render":
                return self._send(200, api_dispersion_render(body))
            if u.path == "/api/focus/setup":
                return self._send(200, api_focus_setup(body))
            if u.path == "/api/focus/bscan":
                return self._send(200, api_focus_bscan(body))
            if u.path == "/api/focus/fit":
                return self._send(200, api_focus_fit(body))
            if u.path == "/api/focus/finish":
                return self._send(200, api_focus_finish(body))
            if u.path == "/api/preview":
                return self._send(200, api_preview(body))
            if u.path == "/api/run":
                JOB.start(body["config"])
                return self._send(200, {"ok": True})
            if u.path == "/api/cancel":
                JOB.cancel.set()
                return self._send(200, {"ok": True})
            if u.path == "/api/open":
                open_in_file_manager(body["path"])
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})


def serve(host="127.0.0.1", port=8765, open_browser=True):
    import socket
    import webbrowser
    for p in range(port, port + 50):          # find a free port
        with socket.socket() as s:
            if s.connect_ex((host, p)) != 0:
                port = p
                break
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"octrecon web UI running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        JOB.cancel.set()
        httpd.server_close()
