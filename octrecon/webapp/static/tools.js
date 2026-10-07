"use strict";
window.__errors = [];
window.addEventListener("error", (e) => window.__errors.push(String(e.message)));
window.addEventListener("unhandledrejection", (e) => window.__errors.push(String(e.reason)));
/* Interactive legacy tools, ported to the web app.
 *  1. Manual dispersion correction  (Demo_DispersionCorrectionManual.m)
 *  2. Choose Focus Positions          (yOCTMeasureFocusDrift.m measureFocus + helpers)
 * Uses helpers/state from app.js ($, api, toast, S, device, dispersionValue, ...). */

// =============================================================== shared: tile list
const T = { vol: null, tiles: null };
async function loadTiles() {
  const vol = S.info.volume_folder;
  if (T.vol === vol && T.tiles) return T.tiles;
  T.tiles = await api("/api/tiles", { volume_folder: vol });
  T.vol = vol;
  T.tiles.byPos = {};
  for (const t of T.tiles.tiles) T.tiles.byPos[`${t.xi},${t.yi},${t.zi}`] = t;
  T.tiles.byFolder = {};
  for (const t of T.tiles.tiles) T.tiles.byFolder[t.folder] = t;
  return T.tiles;
}
const um = (mm) => `${(mm * 1000).toFixed(0)} µm`;

// =============================================================== 1. dispersion tuner
const D = { xi: 0, yi: 0, zi: 0, frame: 1, slider: 2, inflight: false, queued: false, beta: null };
window.octDispState = D;    // exposed for automated UI tests

async function openDispersionTool() {
  if (!S.info) return toast("Inspect a volume first", true);
  try { await loadTiles(); } catch (e) { return toast(e.message, true); }
  const tl = T.tiles;
  // start on the central tile at the depth closest to the tissue surface (z = 0)
  const amin = (a) => a.reduce((b, v, i) => (Math.abs(v) < Math.abs(a[b]) ? i : b), 0);
  if (D.vol !== tl.volume) {
    Object.assign(D, { xi: amin(tl.x_centers_mm), yi: amin(tl.y_centers_mm), zi: amin(tl.z_depths_mm), frame: 1, vol: tl.volume });
  }
  const b0 = dispersionValue();
  D.slider = b0 ? Math.round(Math.sign(b0) * Math.log10(Math.abs(b0)) * 1e6) / 1e6 : 2;   // start at the current value
  buildTileMap(); buildFolderSelect();
  $("d-frame").max = tl.n_frames; $("d-frame-slider").max = tl.n_frames;
  $("dmodal").classList.remove("hidden");
  renderDispersion(true);
}

function buildTileMap() {
  const tl = T.tiles, m = $("d-map");
  const nx = tl.x_centers_mm.length, ny = tl.y_centers_mm.length;
  m.style.gridTemplateColumns = `repeat(${nx}, 1fr)`;
  m.innerHTML = "";
  for (let yi = ny - 1; yi >= 0; yi--) {            // +y up, like a map
    for (let xi = 0; xi < nx; xi++) {
      const c = document.createElement("div");
      c.className = "tile-cell"; c.dataset.xi = xi; c.dataset.yi = yi;
      c.title = `x ${tl.x_centers_mm[xi]} mm, y ${tl.y_centers_mm[yi]} mm`;
      c.onclick = () => { D.xi = xi; D.yi = yi; renderDispersion(true); };
      m.appendChild(c);
    }
  }
}
function buildFolderSelect() {
  const sel = $("d-folder");
  sel.innerHTML = T.tiles.tiles.map((t) => `<option value="${t.folder}">${t.folder} · x ${t.x_mm} · y ${t.y_mm} · z ${um(t.z_mm)}</option>`).join("");
  sel.onchange = () => { const t = T.tiles.byFolder[sel.value]; Object.assign(D, { xi: t.xi, yi: t.yi, zi: t.zi }); renderDispersion(true); };
}
function currentTile() { return T.tiles.byPos[`${D.xi},${D.yi},${D.zi}`]; }

function updateDispersionNav() {
  const tl = T.tiles, t = currentTile();
  $("d-zlab").textContent = `depth ${um(tl.z_depths_mm[D.zi])} (${D.zi + 1}/${tl.z_depths_mm.length})`;
  $("d-xlab").textContent = `x ${tl.x_centers_mm[D.xi]} mm (${D.xi + 1}/${tl.x_centers_mm.length})`;
  $("d-ylab").textContent = `y ${tl.y_centers_mm[D.yi]} mm (${D.yi + 1}/${tl.y_centers_mm.length})`;
  if (t) $("d-folder").value = t.folder;
  $("d-frame").value = D.frame; $("d-frame-slider").value = D.frame;
  for (const c of $("d-map").children) c.classList.toggle("sel", +c.dataset.xi === D.xi && +c.dataset.yi === D.yi);
  $("d-slider").value = D.slider; $("d-slider-val").value = D.slider.toFixed(4);
}

function dispersionNav(axis, delta) {
  const tl = T.tiles;
  const clamp = (v, n) => Math.max(0, Math.min(n - 1, v));
  if (axis === "x") D.xi = clamp(D.xi + delta, tl.x_centers_mm.length);
  else if (axis === "y") D.yi = clamp(D.yi + delta, tl.y_centers_mm.length);
  else if (axis === "z") D.zi = clamp(D.zi + delta, tl.z_depths_mm.length);
  else if (axis === "f") D.frame = Math.max(1, Math.min(tl.n_frames, D.frame + delta));
  renderDispersion(true);
}
function setSlider(v) {
  D.slider = Math.max(-10, Math.min(10, Math.round(v * 1e6) / 1e6));      // 1e-6 in log10 = 2e-6 relative in beta
  renderDispersion(false);
}

// one request in flight; slider moves while busy are coalesced into one follow-up render
async function renderDispersion(newTile) {
  updateDispersionNav();
  if (D.inflight) { D.queued = true; return; }
  const t = currentTile();
  if (!t) return toast("No tile at this position", true);
  D.inflight = true;
  if (newTile) $("d-busy").classList.remove("hidden");
  try {
    const r = await api("/api/dispersion/render", { volume_folder: S.info.volume_folder, folder: t.folder, frame: D.frame,
      slider: D.slider, legacy_axis: $("d-legacy-axis").checked });
    $("d-img").src = "data:image/png;base64," + r.png;
    D.beta = r.beta;
    $("d-title").textContent = `dispersionQuadraticTerm=${r.beta.toExponential(3)} [nm^2/rad]   ·   ${r.folder}, B-scan ${r.frame}` +
      `   (x ${r.tile.x_mm} mm, y ${r.tile.y_mm} mm, z ${um(r.tile.z_mm)})   ·   λ axis: ${r.system_used}`;
    $("d-beta").value = r.beta.toExponential(4);
    $("d-sharp").textContent = `sharpness ${r.sharpness.toFixed(1)} (higher = sharper; guide only)`;
  } catch (e) { toast(e.message, true); }
  D.inflight = false; $("d-busy").classList.add("hidden");
  if (D.queued) { D.queued = false; renderDispersion(false); }
}

function useDispersion() {
  if (D.beta === null) return;
  $("dispersion").value = D.beta.toExponential(4);
  S.touched.dispersion_quadratic_term = "user";
  $("src-dispersion_quadratic_term").textContent = `manual (dispersion tuner: ${currentTile().folder}, B-scan ${D.frame})`;
  $("dmodal").classList.add("hidden");
  toast(`Dispersion set to ${D.beta.toExponential(4)}`);
}

function dispersionKeys(e) {
  if ($("dmodal").classList.contains("hidden")) return;
  if (["INPUT", "SELECT"].includes(document.activeElement.tagName) && document.activeElement.type !== "range") return;
  const step = e.shiftKey ? 0.1 : 0.01;
  if (e.key === "ArrowRight") { setSlider(D.slider + step); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { setSlider(D.slider - step); e.preventDefault(); }
  else if (e.key === "]") dispersionNav("f", 1);
  else if (e.key === "[") dispersionNav("f", -1);
  else if (e.key === "PageUp") { dispersionNav("z", -1); e.preventDefault(); }
  else if (e.key === "PageDown") { dispersionNav("z", 1); e.preventDefault(); }
  else if (e.key === "Escape") $("dmodal").classList.add("hidden");
}

// =============================================================== 2. focus positions
const F = {};   // state of the "Choose Focus Positions" window (mirrors measureFocus)
window.octFocusState = F;   // exposed for automated UI tests

async function openFocusTool() {
  if (!S.info) return toast("Inspect a volume first", true);
  let s;
  try { s = await api("/api/focus/setup", { volume_folder: S.info.volume_folder }); } catch (e) { return toast(e.message, true); }
  Object.assign(F, {
    s, depths: s.depths, n: s.depths.length, k: 0,
    slotPix: Array(s.depths.length).fill(NaN), slotZmm: Array(s.depths.length).fill(NaN),
    xTileI: s.xi0, yIInFile: s.frame_center, img: null, click: null, pred: null, hover: null,
    beta: dispersionValue(),
  });
  $("f-result").classList.add("hidden"); $("f-result").innerHTML = "";
  for (const id of ["f-accept", "f-skip", "f-stop"]) $(id).disabled = false;
  $("f-drift").textContent = "Drift so far: need >= 2 clicked points";
  if (s.shallow_fallback) toast("Shallow scan: no depth >= 50 µm inside the tissue; measuring from z = 0 (legacy fallback).");
  $("fmodal").classList.remove("hidden");
  await loadDepth();
}

function measuredLists(includeK = true) {
  const z = [], p = [];
  F.slotPix.forEach((v, i) => { if (!isNaN(v) && (includeK || i !== F.k)) { z.push(F.depths[i].z_mm); p.push(v); } });
  return { z, p };
}

async function loadDepth() {
  const d = F.depths[F.k];
  F.click = isNaN(F.slotPix[F.k]) ? null : { pix: F.slotPix[F.k], zmm: F.slotZmm[F.k] };   // revisiting shows its click
  const m = measuredLists(true);
  F.pred = null;
  try {
    const r = await api("/api/focus/fit", { volume_folder: S.info.volume_folder, z_stage_mm: m.z, focus_pix: m.p, z_query_mm: d.z_mm });
    F.pred = r.pred_pix;
  } catch (e) { /* no guide */ }
  F.title = `Depth ${F.k + 1} of ${F.n}   (stage z = ${d.z_mm.toFixed(3)} mm)\n` +
    `Click the FOCUS then "Accept focus & Next".   If you cannot see it: try another B-scan or "Skip depth" to keep going without this one, or "Stop measuring here".\n` +
    `Tip: the focus is the bright BAND that stays near the same depth in the X axis.`;
  $("f-title").textContent = F.title;
  $("f-dlab").textContent = `Depth ${F.k + 1} / ${F.n}`;
  await renderFocusBScan();
}

async function renderFocusBScan() {
  const d = F.depths[F.k];
  $("f-busy").classList.remove("hidden");
  try {
    const r = await api("/api/focus/bscan", { volume_folder: S.info.volume_folder, zi: d.zi, xi: F.xTileI,
      frame: F.yIInFile, dispersion_quadratic_term: F.beta });
    if (r.fallback) toast(`No tile at this depth for X tile ${F.xTileI + 1}; using the central tile.`);
    F.xTileI = r.xi; F.folder = r.folder;
    const bytes = Uint8Array.from(atob(r.data_f32), (c) => c.charCodeAt(0));
    F.img = { data: new Float32Array(bytes.buffer), nz: r.shape[0], nx: r.shape[1], lo: r.base_lo, hi: r.base_hi };
    F.beta = r.dispersion;
  } catch (e) { toast(e.message, true); }
  $("f-busy").classList.add("hidden");
  $("f-blab").textContent = `B-scan ${F.yIInFile} / ${F.s.n_frames}`;
  $("f-xlab").textContent = `X tile ${F.xTileI + 1} / ${F.s.x_centers_mm.length}`;
  drawFocus();
}

// ---- drawing (imagesc with CLim from the brightness/contrast sliders, onAdjustDisplay)
const PAD = { l: 70, r: 20, t: 10, b: 46 };
function climFromSliders() {
  const b = +$("f-bright").value, c = +$("f-contrast").value;
  const baseCenter = 0.5 * (F.img.lo + F.img.hi), baseWidth = F.img.hi - F.img.lo;
  const width = baseWidth / c, center = baseCenter - b * 0.5 * baseWidth;
  let lo = center - 0.5 * width, hi = center + 0.5 * width;
  if (hi <= lo) hi = lo + 1e-12;
  return [lo, hi];
}
function plotRect(cv) { return { x: PAD.l, y: PAD.t, w: cv.width - PAD.l - PAD.r, h: cv.height - PAD.t - PAD.b }; }
function zOfPix(p) { const s = F.s; return s.z_mm_first + (p - 1) * (s.z_mm_last - s.z_mm_first) / (s.n_z - 1); }
function yOfPix(p, R) { return R.y + (p - 0.5) / F.img.nz * R.h; }

function drawFocus() {
  const cv = $("f-canvas"), ctx = cv.getContext("2d");
  const W = cv.clientWidth || 1100;
  if (cv.width !== Math.round(W)) { cv.width = Math.round(W); cv.height = Math.round(W * 0.5); }
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--card") || "#fff";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!F.img) return;
  const { nz, nx, data } = F.img, [lo, hi] = climFromSliders();
  const off = document.createElement("canvas"); off.width = nx; off.height = nz;
  const octx = off.getContext("2d"), im = octx.createImageData(nx, nz), sc = 255 / (hi - lo);
  for (let i = 0; i < nz * nx; i++) {
    let v = (data[i] - lo) * sc; v = v < 0 ? 0 : v > 255 ? 255 : v;
    im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255;
  }
  octx.putImageData(im, 0, 0);
  const R = plotRect(cv);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, R.x, R.y, R.w, R.h);
  // axes
  const fg = getComputedStyle(document.body).color;
  ctx.strokeStyle = fg; ctx.fillStyle = fg; ctx.lineWidth = 1; ctx.font = "12px sans-serif";
  ctx.strokeRect(R.x, R.y, R.w, R.h);
  const s = F.s;
  for (let i = 0; i <= 5; i++) {        // x ticks
    const xv = s.x_mm_first + i * (s.x_mm_last - s.x_mm_first) / 5, px = R.x + i / 5 * R.w;
    ctx.beginPath(); ctx.moveTo(px, R.y + R.h); ctx.lineTo(px, R.y + R.h + 4); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(xv.toFixed(2), px, R.y + R.h + 17);
  }
  for (let i = 0; i <= 6; i++) {        // z ticks
    const zv = s.z_mm_first + i * (s.z_mm_last - s.z_mm_first) / 6, py = R.y + i / 6 * R.h;
    ctx.beginPath(); ctx.moveTo(R.x - 4, py); ctx.lineTo(R.x, py); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(zv.toFixed(2), R.x - 6, py + 4);
  }
  ctx.textAlign = "center"; ctx.fillText("x [mm]", R.x + R.w / 2, cv.height - 6);
  ctx.save(); ctx.translate(14, R.y + R.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("z [mm]  (absolute tile depth)", 0, 0); ctx.restore();
  // blue dashed guide (running drift fit)
  if (F.pred !== null && F.pred >= 1 && F.pred <= nz) {
    const y = yOfPix(Math.round(F.pred), R);
    ctx.save(); ctx.setLineDash([8, 6]); ctx.strokeStyle = "rgb(77,204,255)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(R.x, y); ctx.lineTo(R.x + R.w, y); ctx.stroke(); ctx.restore();
  }
  // yellow focus line (click)
  if (F.click) {
    const y = yOfPix(F.click.pix, R);
    ctx.save(); ctx.strokeStyle = "rgb(242,217,38)"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(R.x, y); ctx.lineTo(R.x + R.w, y); ctx.stroke(); ctx.restore();
    ctx.fillStyle = "rgb(242,217,38)"; ctx.textAlign = "left";
    ctx.fillText(`focus ${F.click.pix} px  (z ${F.click.zmm.toFixed(4)} mm)`, R.x + 6, y - 6);
  }
}

function canvasPix(ev) {
  const cv = $("f-canvas"), rect = cv.getBoundingClientRect();
  const sx = cv.width / rect.width, sy = cv.height / rect.height;
  const X = (ev.clientX - rect.left) * sx, Y = (ev.clientY - rect.top) * sy, R = plotRect(cv);
  if (!F.img || X < R.x || X > R.x + R.w || Y < R.y || Y > R.y + R.h) return null;
  // nearest pixel centre, as [~, pix] = min(abs(zClick - z.values)) in onClickFocus
  return Math.max(1, Math.min(F.img.nz, Math.floor((Y - R.y) / R.h * F.img.nz) + 1));
}
function focusClick(ev) {
  const p = canvasPix(ev); if (!p) return;
  F.click = { pix: p, zmm: zOfPix(p) };
  drawFocus();
}
function focusHover(ev) {
  const p = canvasPix(ev);
  $("f-hover").textContent = p ? `cursor: pixel ${p}, z ${zOfPix(p).toFixed(4)} mm` : "";
}

async function updateDriftReadout() {
  const m = measuredLists(true);
  try {
    const r = await api("/api/focus/fit", { volume_folder: S.info.volume_folder, z_stage_mm: m.z, focus_pix: m.p });
    $("f-drift").textContent = r.readout;
  } catch (e) { /* keep text */ }
}

async function focusAction(action) {
  if (action === "next") {
    if (!F.click) { toast(`No focus click registered for this depth; skipping.`, true); }
    else { F.slotPix[F.k] = F.click.pix; F.slotZmm[F.k] = F.click.zmm; await updateDriftReadout(); }
    F.k += 1;
  } else if (action === "skip") {
    F.slotPix[F.k] = NaN; F.slotZmm[F.k] = NaN; await updateDriftReadout(); F.k += 1;
  } else if (action === "stop") {
    return finishFocus();
  }
  if (F.k >= F.n) return finishFocus();
  await loadDepth();
}

async function focusNav(kind, delta) {
  if (kind === "b") {
    const v = Math.max(1, Math.min(F.s.n_frames, F.yIInFile + delta)); if (v === F.yIInFile) return;
    F.yIInFile = v; await renderFocusBScan();
  } else if (kind === "x") {
    const v = Math.max(0, Math.min(F.s.x_centers_mm.length - 1, F.xTileI + delta)); if (v === F.xTileI) return;
    if (!F.s.folders[`${F.depths[F.k].zi},${v}`]) return toast(`No tile at this depth for X tile ${v + 1}; staying on tile ${F.xTileI + 1}.`, true);
    F.xTileI = v; await renderFocusBScan();
  } else if (kind === "d") {        // navigate without recording (only Accept/Skip modify a depth)
    const v = Math.max(0, Math.min(F.n - 1, F.k + delta)); if (v === F.k) return;
    F.k = v; await loadDepth();
  }
}

async function finishFocus() {
  const meas = [];
  F.slotPix.forEach((v, i) => { if (!isNaN(v)) meas.push({ zi: F.depths[i].zi, pix: v, z_mm: F.slotZmm[i] }); });
  for (const id of ["f-accept", "f-skip", "f-stop"]) $(id).disabled = true;
  const out = $("f-result");
  if (!meas.length) {
    out.innerHTML = `<div class="note warn">No focus measurements were recorded (the focus was not accepted on any depth). Nothing was saved.</div>`;
    out.classList.remove("hidden"); return;
  }
  const outDir = ($("output_root").value && $("output_name").value) ? joinPath($("output_root").value, $("output_name").value) : null;
  try {
    const r = await api("/api/focus/finish", { volume_folder: S.info.volume_folder, measurements: meas,
      out_dir: outDir, also_volume: $("f-also-volume").checked });
    F.result = r;
    const d = r.diagnostics;
    out.innerHTML = `<div class="note ok"><b>Focus positions for all ${r.focus_positions.length} depths:</b> ${r.focus_positions.map((x) => Math.round(x)).join(", ")} px
      <br>Drift slope ${d.driftSlope.toFixed(4)} um/um · tissue RI ${typeof d.tissueRI === "number" ? d.tissueRI.toFixed(3) : d.tissueRI} · regime ${d.driftRegime}
      ${(d.messages || []).length ? "<br>" + [].concat(d.messages).join("<br>") : ""}
      <br>Saved: ${r.saved.map((p) => `<code>${p}</code>`).join(", ")}</div>
      <div class="row"><img class="drift-fig" src="data:image/png;base64,${r.figure_png}">
      <button class="primary" id="f-use">Use these focus positions</button></div>`;
    out.classList.remove("hidden");
    $("f-use").onclick = () => {
      $("focus_mode").value = "file"; onFocusMode(); $("focus_file").value = r.saved[0];
      $("src-focus").textContent = `Choose Focus Positions tool: ${summariseFocus(r.focus_positions)}`;
      $("fmodal").classList.add("hidden"); toast("Focus positions set from the focus tool");
    };
  } catch (e) { out.innerHTML = `<div class="note warn">${e.message}</div>`; out.classList.remove("hidden"); }
}

// =============================================================== wiring
window.addEventListener("DOMContentLoaded", () => {
  $("btn-tune-disp").onclick = openDispersionTool;
  $("dmodal-close").onclick = () => $("dmodal").classList.add("hidden");
  document.querySelectorAll("[data-dnav]").forEach((b) => { const [a, d] = b.dataset.dnav.split(","); b.onclick = () => dispersionNav(a, +d); });
  document.querySelectorAll("[data-dstep]").forEach((b) => (b.onclick = () => setSlider(D.slider + (+b.dataset.dstep))));
  $("d-slider").oninput = (e) => setSlider(+e.target.value);
  $("d-slider-val").onchange = (e) => setSlider(+e.target.value);
  $("d-beta").onchange = (e) => { const b = Number(e.target.value); if (Number.isFinite(b) && b !== 0) setSlider(Math.sign(b) * Math.log10(Math.abs(b))); };
  $("d-frame").onfocus = (e) => e.target.select();
  $("d-frame").onchange = (e) => { D.frame = Math.max(1, Math.min(T.tiles.n_frames, +e.target.value || 1)); renderDispersion(true); };
  $("d-frame-slider").oninput = (e) => { D.frame = +e.target.value; renderDispersion(true); };
  $("d-legacy-axis").onchange = () => renderDispersion(true);
  $("d-use").onclick = useDispersion;
  $("d-reset").onclick = () => { const a = S.auto.dispersion_quadratic_term; if (a) setSlider(Math.sign(a.value) * Math.log10(Math.abs(a.value))); };
  document.addEventListener("keydown", dispersionKeys);

  $("btn-focus-tool").onclick = openFocusTool;
  $("fmodal-close").onclick = () => $("fmodal").classList.add("hidden");
  $("f-canvas").addEventListener("click", focusClick);
  $("f-canvas").addEventListener("mousemove", focusHover);
  $("f-bright").oninput = $("f-contrast").oninput = () => drawFocus();
  $("f-accept").onclick = () => focusAction("next");
  $("f-skip").onclick = () => focusAction("skip");
  $("f-stop").onclick = () => focusAction("stop");
  document.querySelectorAll("[data-fnav]").forEach((b) => { const [k, d] = b.dataset.fnav.split(","); b.onclick = () => focusNav(k, +d); });
  window.addEventListener("resize", () => { if (!$("fmodal").classList.contains("hidden")) drawFocus(); });
});
