"use strict";
const $ = (id) => document.getElementById(id);
const S = { sys: null, info: null, auto: {}, touched: {}, logN: 0, poll: null, sep: "/" };

// ------------------------------------------------------------------ utils
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function toast(msg, err = false) {
  const t = $("toast"); t.textContent = msg; t.className = "toast" + (err ? " err" : "");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), err ? 8000 : 3500);
}
function busy(btn, on) { btn.disabled = on; btn.classList.toggle("spin", on); }
function fmtNum(v) { return (v === null || v === undefined) ? "" : (Math.abs(v) >= 1e5 ? Number(v).toExponential(4).replace("e+", "e") : String(v)); }
function joinPath(a, b) { if (!a) return b; return a.replace(/[\\/]+$/, "") + S.sep + b; }
function parseNum(s) { const v = Number(String(s).trim()); return Number.isFinite(v) ? v : null; }
function parseRows(s) {
  s = s.trim(); if (!s) return null;
  const out = [];
  for (const part of s.split(",")) {
    const m = part.trim().match(/^(\d+)\s*-\s*(\d+)$/);
    if (m) for (let i = +m[1]; i <= +m[2]; i++) out.push(i);
    else if (part.trim()) out.push(parseInt(part, 10));
  }
  return out.filter(Number.isFinite);
}
function save() { try { localStorage.setItem("octrecon", JSON.stringify({ volume: $("volume").value, output_root: $("output_root").value })); } catch (e) {} }

// ------------------------------------------------------------------ system
async function loadSystem() {
  const s = S.sys = await api("/api/system");
  S.sep = s.platform.toLowerCase().includes("windows") ? "\\" : "/";
  const b = $("sysbadges"); b.innerHTML = "";
  const add = (txt, cls) => { const e = document.createElement("span"); e.className = "badge " + (cls || ""); e.textContent = txt; b.appendChild(e); };
  add(s.gpu_available ? `GPU: ${s.gpu_name}${s.gpu_kind === "mps" ? "" : ` (${s.vram_GB} GB)`}` : "GPU: not available", s.gpu_available ? "ok" : "warn");
  add(`CPU: ${s.cpu_count} threads`);
  if (s.ram_total_GB) add(`RAM: ${s.ram_available_GB}/${s.ram_total_GB} GB free`, s.ram_available_GB < 5 ? "warn" : "");
  if (s.ram_available_GB && s.ram_available_GB < 5)
    toast(`Only ${s.ram_available_GB} GB RAM free: a reconstruction needs ~3-6 GB. Close large programs (e.g. image viewers) before starting.`, true);
  $("gpu-desc").textContent = !s.gpu_available ? (s.gpu_note || "not available")
    : s.gpu_kind === "mps" ? `${s.gpu_name} · PyTorch Metal backend (float32)` : `${s.gpu_name} · CuPy/CUDA · fastest (I/O-bound)`;
  $("cpu-desc").textContent = `${s.cpu_count} threads · Numba + multithreaded FFT (~5× slower than GPU)`;
  if (!s.gpu_available) { $("dev-gpu").classList.add("disabled"); document.querySelector("input[value=gpu]").disabled = true; }
  document.querySelector(`input[value=${s.gpu_available ? "gpu" : "cpu"}]`).checked = true;
  const d = s.defaults;
  for (const k of ["batch_frames", "io_threads", "prefetch_batches"]) $(k).value = d[k];
  $("precision").value = d.precision;
  for (const k of ["gpu_fused_kernel", "write_tiff", "keep_float_volume", "legacy_double_quantization"]) $(k).checked = !!d[k];
}

// ------------------------------------------------------------------ inspect
const AUTO_FIELDS = { dispersion_quadratic_term: "dispersion", focus_sigma: "focus_sigma", n_medium: "n_medium",
                      crop_z_range_mm: "crop", output_pixel_size_um: "pixel" };
async function inspect() {
  const btn = $("btn-inspect"); busy(btn, true);
  try {
    const info = S.info = await api("/api/inspect", { volume_folder: $("volume").value.trim() });
    $("volume").value = info.volume_folder;
    const sc = info.scan, st = info.storage;
    const kv = (v, k) => `<div class="kv"><b>${v}</b><span>${k}</span></div>`;
    $("scan-summary").innerHTML =
      kv(`${sc.oct_system}`, "OCT system") + kv(sc.probe || "–", "probe / objective") +
      kv(`${sc.grid_xyz[0]}×${sc.grid_xyz[1]}×${sc.grid_xyz[2]}`, `tiles x×y×depth (${sc.n_tiles})`) +
      kv(`${sc.tile_px[0]}×${sc.tile_px[1]} px`, `per tile · ${sc.pixel_size_um} µm`) +
      kv(`${sc.raw_size_GB} GB`, "raw spectra") + kv(`${sc.spectral_px}×${sc.alines_per_bscan}`, `spectral px × A-lines (${sc.apod_lines} apod.)`) +
      kv(`${sc.bscan_avg}/${sc.ascan_avg}/${sc.spectra_avg}`, "B-scan / A-scan / spectral avg") +
      kv(sc.estimated_output_shape_yzx.join("×"), "output ≈ y×z×x voxels");
    const n = $("storage-note");
    const c = st.counts;
    if (st.n_missing) { n.className = "note warn"; n.textContent = `⚠ ${st.n_missing} tile folders are missing or unreadable (e.g. ${st.missing.slice(0, 3).join(", ")}).`; }
    else if (st.layout === "unzipped") { n.className = "note ok"; n.textContent = `✓ All ${c.unzipped} tiles are unzipped (Header.xml + data/Spectral*.data).`; }
    else { n.className = "note"; n.textContent = `${c.oct} tiles are compressed .oct archives${c.unzipped ? `, ${c.unzipped} unzipped` : ""}. They are read in place by default (Advanced → extract like legacy).`; }
    $("inspect-out").classList.remove("hidden");
    // auto values
    S.auto = info.auto; S.touched = {};
    for (const [key, field] of Object.entries(AUTO_FIELDS)) {
      const a = info.auto[key];
      $(field).value = key === "crop_z_range_mm" ? (a.value ? a.value.join(", ") : "") : fmtNum(a.value);
      $("src-" + key).textContent = a.source;
    }
    const f = info.auto.focus_positions;
    $("focus_mode").value = "auto"; onFocusMode();
    $("src-focus").textContent = f.value ? `${f.source}: ${summariseFocus(f.value)}` : f.source;
    // output defaults
    const parts = info.volume_folder.split(/[\\/]/).filter(Boolean);
    const isOCT = parts[parts.length - 1].toLowerCase() === "octvolume";
    const sample = isOCT ? parts[parts.length - 2] : parts[parts.length - 1];
    const parent = info.volume_folder.replace(/[\\/]+$/, "").split(/[\\/]/).slice(0, isOCT ? -1 : -1).join(S.sep) || S.sep;
    if (!$("output_root").value) $("output_root").value = joinPath(parent, "reconstructions");
    if (!$("output_name").value || S._autoName === $("output_name").value) { $("output_name").value = S._autoName = `${sample}_recon`; }
    updatePath(); save();
    toast("Scan inspected — parameters filled in automatically");
  } catch (e) { toast(e.message, true); }
  busy(btn, false);
}
function summariseFocus(v) { const u = [...new Set(v.map((x) => (x === null ? "none" : Math.round(x))))]; return u.length === 1 ? `${u[0]} px (all depths)` : v.map((x) => (x === null ? "–" : Math.round(x))).join(", "); }
function updatePath() { $("final-path").textContent = joinPath(joinPath($("output_root").value, $("output_name").value), `${$("output_name").value}.tiff`); }

// ------------------------------------------------------------------ focus
function onFocusMode() {
  const m = $("focus_mode").value;
  $("focus-file-row").classList.toggle("hidden", m !== "file");
  $("focus-value-row").classList.toggle("hidden", m !== "value");
}
async function detectFocus() {
  if (!S.info) return toast("Inspect a volume first", true);
  const btn = $("btn-detect-focus"); busy(btn, true); $("focus-status").textContent = "reading a few B-scans…";
  try {
    const r = await api("/api/estimate/focus", { volume_folder: S.info.volume_folder, device: device(), dispersion_quadratic_term: dispersionValue() });
    $("focus_mode").value = "value"; onFocusMode();
    $("focus_value").value = r.focus_positions.map((x) => Math.round(x)).join(", ");
    $("src-focus").textContent = `automatic detection (${r.method || ""}) · ${summariseFocus(r.focus_positions)}`;
    $("focus-status").textContent = `done in ${(r.elapsed_s || 0).toFixed(1)} s`;
    showFig("focus-fig", r.figure_png);
  } catch (e) { toast(e.message, true); $("focus-status").textContent = ""; }
  busy(btn, false);
}
function showFig(id, b64) { const el = $(id); if (!b64) return el.classList.add("hidden"); el.innerHTML = `<img src="data:image/png;base64,${b64}">`; el.classList.remove("hidden"); }

function currentFocus() {
  const m = $("focus_mode").value;
  if (m === "value") { const v = $("focus_value").value.split(/[,\s]+/).map(parseNum).filter((x) => x !== null); if (!v.length) return null;
    const n = S.info ? S.info.scan.z_depths_mm.length : v.length; return v.length === 1 ? Array(n).fill(v[0]) : v; }
  if (m === "auto" && S.auto.focus_positions && S.auto.focus_positions.value) return S.auto.focus_positions.value;
  return null;
}

// ------------------------------------------------------------------ dispersion
function dispersionValue() { return parseNum($("dispersion").value); }
async function estimateDispersion() {
  if (!S.info) return toast("Inspect a volume first", true);
  const btn = $("btn-est-disp"); busy(btn, true);
  try {
    const r = await api("/api/estimate/dispersion", { volume_folder: S.info.volume_folder, device: device(), initial: dispersionValue() });
    $("dispersion").value = fmtNum(r.value); S.touched.dispersion_quadratic_term = "estimate";
    $("src-dispersion_quadratic_term").textContent = `automatic estimate (${r.method || ""}) in ${(r.elapsed_s || 0).toFixed(0)} s`;
    showFig("disp-fig", r.figure_png);
  } catch (e) { toast(e.message, true); }
  busy(btn, false);
}

// ------------------------------------------------------------------ config
function device() { const r = document.querySelector("input[name=device]:checked"); return r ? r.value : "auto"; }
function buildConfig() {
  if (!$("volume").value.trim()) throw new Error("Select the raw OCT volume folder first (step 1)");
  if (!$("output_root").value.trim() || !$("output_name").value.trim()) throw new Error("Set the output folder and reconstruction name (step 2)");
  const val = (key, field, conv) => (S.touched[key] ? conv($(field).value) : "auto");
  const cfg = {
    volume_folder: $("volume").value.trim(),
    output_root: $("output_root").value.trim(), output_name: $("output_name").value.trim(),
    dispersion_quadratic_term: val("dispersion_quadratic_term", "dispersion", parseNum),
    focus_sigma: val("focus_sigma", "focus_sigma", parseNum),
    n_medium: val("n_medium", "n_medium", parseNum),
    output_pixel_size_um: val("output_pixel_size_um", "pixel", parseNum),
    crop_z_range_mm: val("crop_z_range_mm", "crop", (s) => { const v = s.split(/[,\s]+/).map(parseNum).filter((x) => x !== null); return v.length === 2 ? v : "none"; }),
    interp_method: $("interp_method").value,
    apply_path_length_correction: $("opc").checked,
    raw_input: $("raw_input").value, delete_archives_after_extract: $("delete_archives").checked,
    device: device(), precision: $("precision").value, gpu_fused_kernel: $("gpu_fused_kernel").checked,
    batch_frames: +$("batch_frames").value, io_threads: +$("io_threads").value, prefetch_batches: +$("prefetch_batches").value,
    rows: parseRows($("rows").value), write_tiff: $("write_tiff").checked, keep_float_volume: $("keep_float_volume").checked,
    legacy_double_quantization: $("legacy_double_quantization").checked,
  };
  const m = $("focus_mode").value;
  if (m === "file") cfg.focus_positions = $("focus_file").value.trim();
  else if (m === "value") { const f = currentFocus(); if (!f) throw new Error("Enter focus value(s)"); cfg.focus_positions = f; }
  else cfg.focus_positions = m; // auto | estimate | none
  for (const [k, v] of Object.entries(cfg)) if (v === null && !["rows"].includes(k)) throw new Error(`Invalid value for ${k}`);
  return cfg;
}
function applyConfig(c) {
  $("volume").value = c.volume_folder || ""; $("output_root").value = c.output_root || ""; $("output_name").value = c.output_name || "";
  const setAuto = (key, field, fmt) => { if (c[key] !== undefined && c[key] !== "auto") { $(field).value = fmt(c[key]); S.touched[key] = "user"; $("src-" + key).textContent = "user (loaded config)"; } };
  setAuto("dispersion_quadratic_term", "dispersion", fmtNum); setAuto("focus_sigma", "focus_sigma", fmtNum); setAuto("n_medium", "n_medium", fmtNum);
  setAuto("output_pixel_size_um", "pixel", fmtNum); setAuto("crop_z_range_mm", "crop", (v) => (Array.isArray(v) ? v.join(", ") : ""));
  if (Array.isArray(c.focus_positions) || typeof c.focus_positions === "number") { $("focus_mode").value = "value"; $("focus_value").value = [].concat(c.focus_positions).join(", "); }
  else if (typeof c.focus_positions === "string" && c.focus_positions.endsWith(".mat")) { $("focus_mode").value = "file"; $("focus_file").value = c.focus_positions; }
  else if (c.focus_positions) $("focus_mode").value = c.focus_positions;
  onFocusMode();
  for (const k of ["precision", "interp_method", "raw_input"]) if (c[k]) $(k).value = c[k];
  for (const k of ["batch_frames", "io_threads", "prefetch_batches"]) if (c[k]) $(k).value = c[k];
  for (const k of ["gpu_fused_kernel", "write_tiff", "keep_float_volume", "legacy_double_quantization"]) if (c[k] !== undefined) $(k).checked = !!c[k];
  if (c.apply_path_length_correction !== undefined) $("opc").checked = !!c.apply_path_length_correction;
  if (c.rows) $("rows").value = c.rows.join(",");
  if (c.device) { const r = document.querySelector(`input[value=${c.device}]`); if (r && !r.disabled) r.checked = true; }
  updatePath();
}

// ------------------------------------------------------------------ run
async function run() {
  let cfg; try { cfg = buildConfig(); } catch (e) { return toast(e.message, true); }
  try { await api("/api/run", { config: cfg }); } catch (e) { return toast(e.message, true); }
  S.logN = 0; $("log").textContent = ""; $("log").classList.remove("hidden"); $("result").classList.add("hidden");
  $("progress-wrap").classList.remove("hidden"); $("btn-run").disabled = true; $("btn-cancel").classList.remove("hidden");
  startPolling();
}
function startPolling() { clearInterval(S.poll); S.poll = setInterval(pollJob, 1000); pollJob(); }
async function pollJob() {
  let j; try { j = await api(`/api/job?since=${S.logN}`); } catch (e) { return; }
  if (j.logs.length) { const L = $("log"); L.textContent += j.logs.join("\n") + "\n"; L.scrollTop = L.scrollHeight; S.logN = j.n_logs; L.classList.remove("hidden"); }
  const p = j.progress || {};
  let pct = 0, txt = j.state;
  if (p.stage === "reconstruct") { pct = p.done; txt = `Reconstructing · row ${p.row}/${p.rows} · ${pct.toFixed(1)} %` +
      (p.elapsed_s ? ` · elapsed ${fmtT(p.elapsed_s)}` : "") + (p.eta_s ? ` · ETA ${fmtT(p.eta_s)}` : ""); }
  else if (p.stage === "extract") { pct = 100 * p.done / p.total; txt = `Extracting .oct archives ${p.done}/${p.total}`; }
  else if (p.stage === "write_tiff") { pct = 100 * p.done / p.total; txt = `Writing BigTIFF ${p.done}/${p.total} planes`; }
  else if (j.state === "starting") txt = "Preparing (reading headers, building operators)…";
  if (j.state === "done") pct = 100;
  $("progress-bar").style.width = `${pct}%`; $("progress-text").textContent = txt;
  if (["done", "error", "cancelled", "idle"].includes(j.state)) {
    clearInterval(S.poll); $("btn-run").disabled = false; $("btn-cancel").classList.add("hidden");
    if (j.state !== "idle") showResult(j);
  }
}
function fmtT(s) { s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60; return h ? `${h} h ${m} min` : m ? `${m} min ${ss} s` : `${ss} s`; }
function showResult(j) {
  const r = $("result"); r.classList.remove("hidden");
  if (j.state === "done") {
    const s = j.summary;
    r.innerHTML = `<div class="note ok"><b>✓ Reconstruction finished</b> in ${fmtT(s.reconstruction_seconds + (s.tiff_seconds || 0))} on ${s.device.toUpperCase()} ·
      output ${s.output_shape_yzx.join("×")} (y×z×x) · clim [${s.clim_dB.map((x) => x.toFixed(2)).join(", ")}] dB<br>
      <code>${s.tiff || s.output_dir}</code></div>
      <div class="row" style="margin-top:8px"><button class="secondary" id="btn-open">Open output folder</button></div>`;
    $("btn-open").onclick = () => api("/api/open", { path: s.output_dir }).catch((e) => toast(e.message, true));
  } else if (j.state === "error") r.innerHTML = `<div class="note warn"><b>Error:</b> ${j.error}</div>`;
  else r.innerHTML = `<div class="note warn">Cancelled.</div>`;
}

// ------------------------------------------------------------------ folder browser
let B = { target: null, mode: "dir", path: null };
function openBrowser(target, mode) {
  B = { target, mode, path: $(target).value.trim() || (target === "output_root" && S.info ? S.info.volume_folder : "") };
  $("modal-title").textContent = mode === "dir" ? "Select folder" : "Select file";
  $("modal-select").classList.toggle("hidden", mode !== "dir");
  $("modal").classList.remove("hidden"); loadDir(B.path);
}
async function loadDir(path) {
  try {
    const d = await api("/api/browse?path=" + encodeURIComponent(path || ""));
    B.path = d.path; $("modal-path").value = d.path; B.parent = d.parent; B.home = d.home;
    const roots = $("modal-roots"); roots.innerHTML = '<option value="">Go to…</option>' + d.roots.map((r) => `<option>${r}</option>`).join("");
    const ul = $("modal-list"); ul.innerHTML = "";
    for (const e of d.entries) {
      if (B.mode === "dir" && e.type !== "dir") continue;
      const li = document.createElement("li");
      li.className = e.is_volume ? "vol" : "";
      li.innerHTML = `${e.type === "dir" ? "📁" : "📄"} <span>${e.name}</span>${e.is_volume ? '<span class="tag">OCT volume</span>' : e.has_volume ? '<span class="tag">contains OCTVolume</span>' : ""}`;
      li.onclick = () => (e.type === "dir" ? loadDir(e.path) : pick(e.path));
      ul.appendChild(li);
    }
    $("modal-note").textContent = d.is_volume ? "✓ this folder is an OCT volume (ScanInfo.json)" : "";
  } catch (e) { toast(e.message, true); }
}
function pick(p) {
  $(B.target).value = p; $("modal").classList.add("hidden"); updatePath(); save();
  if (B.target === "volume") inspect();
}

// ------------------------------------------------------------------ wiring
window.addEventListener("DOMContentLoaded", async () => {
  try { const st = JSON.parse(localStorage.getItem("octrecon") || "{}"); if (st.volume) $("volume").value = st.volume; if (st.output_root) $("output_root").value = st.output_root; } catch (e) {}
  document.querySelectorAll("[data-browse]").forEach((b) => (b.onclick = () => openBrowser(b.dataset.browse, b.dataset.mode)));
  $("btn-inspect").onclick = inspect;
  $("volume").addEventListener("keydown", (e) => { if (e.key === "Enter") inspect(); });
  $("output_root").oninput = $("output_name").oninput = () => { updatePath(); save(); };
  $("focus_mode").onchange = onFocusMode;
  $("btn-detect-focus").onclick = detectFocus;
  $("btn-est-disp").onclick = estimateDispersion;
  for (const [key, field] of Object.entries(AUTO_FIELDS)) $(field).addEventListener("input", () => { S.touched[key] = "user"; $("src-" + key).textContent = "user"; });
  $("btn-run").onclick = run; $("btn-cancel").onclick = () => api("/api/cancel", {}).then(() => toast("Cancelling…"));
  $("btn-save-cfg").onclick = () => { try { const c = buildConfig(); const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify(c, null, 2)], { type: "application/json" })); a.download = `${c.output_name}_config.json`; a.click(); } catch (e) { toast(e.message, true); } };
  $("load-cfg").onchange = async (e) => { const f = e.target.files[0]; if (!f) return; try { const c = JSON.parse(await f.text()); applyConfig(c.config || c); if (c.volume_folder || (c.config && c.config.volume_folder)) inspect(); toast("Config loaded"); } catch (err) { toast("Invalid config: " + err.message, true); } };
  $("modal-close").onclick = () => $("modal").classList.add("hidden");
  $("modal-up").onclick = () => B.parent && loadDir(B.parent); $("modal-home").onclick = () => loadDir(B.home);
  $("modal-roots").onchange = (e) => e.target.value && loadDir(e.target.value);
  $("modal-path").addEventListener("keydown", (e) => { if (e.key === "Enter") loadDir(e.target.value); });
  $("modal-select").onclick = () => pick(B.path);
  try { await loadSystem(); } catch (e) { toast("Cannot reach the local server: " + e.message, true); }
  try { const j = await api("/api/job?since=0"); if (["starting", "running"].includes(j.state)) { $("progress-wrap").classList.remove("hidden"); $("btn-run").disabled = true; $("btn-cancel").classList.remove("hidden"); startPolling(); } } catch (e) {}
  if ($("volume").value) inspect();
  updatePath();
});
