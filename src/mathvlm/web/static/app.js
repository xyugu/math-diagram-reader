// The viewer page: pick a run and a sample, see the input next to the predicted diagram, drag
// points (snapped to the integer grid) and circle rims to fix it, or edit the JSON directly.
// Every change is re-validated by the server (/api/check); nothing is saved server-side.

import { dist, isObj, paint } from "./diagram.js";

const $ = (id) => document.getElementById(id);
const HANDLE_R = 2.2; // grid units: how close a pointer must be to grab a point
const EDIT_DEBOUNCE_MS = 300;
const MAX_LISTED = 8; // errors or warnings shown before the rest fold away

const state = {
  runs: [],
  run: null, // run key
  samples: [], // SampleSummary[], dataset order
  visible: [], // after filter and sort
  sid: null,
  sample: null, // Sample
  original: "null", // JSON of the prediction as loaded, to tell whether it was edited
  obj: null, // the output being edited (decoded JSON; null when the raw output did not parse)
  history: [], // JSON snapshots for undo
  check: null, // CheckResult of `checkedFor`
  checkedFor: null,
  jsonError: null, // why the JSON box does not parse, while it does not
  drag: null,
  showGt: false,
};

let G; // geometry and style constants, from /api/meta

// -- helpers ----------------------------------------------------------------------------------

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const snapshot = () => JSON.stringify(state.obj);
const diagramOf = (obj) => (isObj(obj) && isObj(obj.diagram) ? obj.diagram : null);

async function api(path, body) {
  const init =
    body === undefined
      ? {}
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, init);
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 400)}`);
  return res;
}
const getJson = async (path) => (await api(path)).json();
const q = (params) => new URLSearchParams(params).toString();

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function geometry(meta) {
  const [gmin, gmax] = meta.grid;
  const r = meta.render;
  const lo = gmin - r.diagram_pad;
  const span = gmax - gmin + 2 * r.diagram_pad;
  const ppg = (r.figure_width_in * 72) / span; // points per grid unit, as in render.py
  return {
    gmin,
    gmax,
    lo,
    span,
    ink: r.ink,
    fontGrid: r.label_fontsize / ppg, // label font size in grid units
    style: {
      // render.default_style(), converted from points to grid units
      ms: r.mark_size,
      lw: r.line_width / ppg,
      markLw: Math.max(0.6, 0.8 * r.line_width) / ppg,
      dotR: r.point_marker_size / 2 / ppg,
      labelDist: (r.label_offset_pt + 0.35 * r.label_fontsize) / ppg,
      tol: meta.collinear_tol,
      pointDots: true,
      angleArcs: true,
      unnamed: new RegExp(`^${escRe(meta.unnamed_prefix)}(?:0|[1-9][0-9]*)$`),
      measure,
    },
  };
}

// -- KaTeX ------------------------------------------------------------------------------------

const texCache = new Map();
function tex(s) {
  let html = texCache.get(s);
  if (html === undefined) {
    html = katex.renderToString(s, { throwOnError: false, output: "html" });
    texCache.set(s, html);
  }
  return html;
}

const pxPerGrid = () => $("svg").clientWidth / G.span;

// Half extents of a label in grid units, background padding included (render.py's
// _half_extent). Measured once per text and stage size.
let extents = { k: 0, map: new Map() };
function measure(s) {
  const k = pxPerGrid();
  if (k <= 0) return [0, 0];
  if (extents.k !== k) extents = { k, map: new Map() };
  let e = extents.map.get(s);
  if (!e) {
    const m = $("measure");
    m.innerHTML = tex(s);
    const r = m.getBoundingClientRect();
    const pad = 0.08 * G.fontGrid * k;
    e = [(r.width / 2 + pad) / k, (r.height / 2 + pad) / k];
    extents.map.set(s, e);
  }
  return e;
}

// Question Markdown: the subset render.py draws (paragraphs, line breaks, $...$, $$...$$).
function renderQuestion(el, text) {
  el.replaceChildren();
  if (typeof text !== "string") {
    el.innerHTML = '<p class="muted">(no text)</p>';
    return;
  }
  if (!text.trim()) {
    el.innerHTML = '<p class="muted">(empty)</p>';
    return;
  }
  for (const para of text.trim().split(/\n[ \t]*\n/)) {
    const p = document.createElement("p");
    para.split(/(\$\$[\s\S]+?\$\$)/).forEach((piece, i) => {
      if (i % 2) {
        p.append(piece); // display math may span lines
        return;
      }
      piece.split("\n").forEach((line, j) => {
        if (j) p.append(document.createElement("br"));
        p.append(line);
      });
    });
    el.append(p);
  }
  renderMathInElement(el, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
}

// -- drawing ----------------------------------------------------------------------------------

function buildGrid() {
  const svg = $("svg");
  svg.setAttribute("viewBox", `${G.lo} ${G.lo} ${G.span} ${G.span}`);
  let s = "";
  for (let v = G.gmin; v <= G.gmax; v += 10) {
    s += `<line x1="${v}" y1="${G.gmin}" x2="${v}" y2="${G.gmax}"/><line x1="${G.gmin}" y1="${v}" x2="${G.gmax}" y2="${v}"/>`;
  }
  const w = G.gmax - G.gmin;
  $("grid").innerHTML = `${s}<rect x="${G.gmin}" y="${G.gmin}" width="${w}" height="${w}"/>`;
}

let frame = 0;
function redraw() {
  if (!frame) {
    frame = requestAnimationFrame(() => {
      frame = 0;
      draw();
    });
  }
}

function emptyMessage() {
  if (!state.sample) return "";
  if (state.obj === null) return "The output is not JSON. Fix it in the box below to draw it.";
  if (!isObj(state.obj)) return "The output is JSON, but not an object.";
  if (state.obj.diagram === null) return 'No diagram: the output says "diagram": null.';
  if (!isObj(state.obj.diagram)) return "No diagram object in the output.";
  return "";
}

function draw() {
  const k = pxPerGrid();
  $("labels").style.fontSize = `${G.fontGrid * k}px`;
  $("labels").style.color = G.ink;
  let svg = "";
  let texts = [];

  const gt = state.showGt ? diagramOf(state.sample?.gt) : null;
  if (gt) svg += `<g class="ink gt">${paint(gt, G.style).svg}</g>`;

  const d = diagramOf(state.obj);
  if (d) {
    const out = paint(d, G.style);
    svg += `<g class="ink pred" style="color:${esc(G.ink)}">${out.svg}</g>`;
    texts = out.texts;
    svg += handlesSvg(out);
  }
  $("layer").innerHTML = svg;
  $("label-list").innerHTML = texts
    .map(
      (t) =>
        `<div class="lbl${t.box ? " box" : ""}" style="left:${(t.x - G.lo) * k}px;top:${(t.y - G.lo) * k}px">${tex(t.tex)}</div>`,
    )
    .join("");

  const msg = d ? "" : emptyMessage();
  $("empty").hidden = !msg;
  $("empty").textContent = msg;
}

function handlesSvg(out) {
  const dr = state.drag;
  let s = "";
  for (const c of out.circles) {
    const [cx, cy] = out.P[c.center];
    const active = dr?.circle === c.index ? " active" : "";
    s += `<circle class="rim${active}" data-circle="${c.index}" cx="${cx}" cy="${cy}" r="${c.radius}"><title>circle ${esc(c.center)}, radius ${c.radius}</title></circle>`;
  }
  for (const [name, [x, y]] of Object.entries(out.P)) {
    const active = dr?.point === name ? " active" : "";
    s += `<circle class="handle${active}" data-point="${esc(name)}" cx="${x}" cy="${y}" r="${HANDLE_R}"><title>${esc(name)} (${x}, ${y})</title></circle>`;
  }
  return s;
}

// -- panels -----------------------------------------------------------------------------------

// One element per line inside the diagram, so it stays readable and editable in place.
function inline(v) {
  if (Array.isArray(v)) return `[${v.map(inline).join(", ")}]`;
  if (isObj(v)) return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${inline(x)}`).join(", ")}}`;
  return JSON.stringify(v);
}

function formatJson(obj) {
  if (!diagramOf(obj)) return JSON.stringify(obj, null, 2);
  const lines = ["{"];
  const top = Object.entries(obj);
  top.forEach(([k, v], i) => {
    const comma = i < top.length - 1 ? "," : "";
    if (k !== "diagram") {
      lines.push(`  ${JSON.stringify(k)}: ${JSON.stringify(v)}${comma}`);
      return;
    }
    lines.push('  "diagram": {');
    const entries = Object.entries(v);
    entries.forEach(([dk, dv], j) => {
      const c2 = j < entries.length - 1 ? "," : "";
      let items = null;
      if (isObj(dv)) items = Object.entries(dv).map(([n, p]) => `${JSON.stringify(n)}: ${inline(p)}`);
      else if (Array.isArray(dv)) items = dv.map(inline);
      if (!items || items.length === 0) {
        lines.push(`    ${JSON.stringify(dk)}: ${inline(dv)}${c2}`);
        return;
      }
      const [open, close] = isObj(dv) ? ["{", "}"] : ["[", "]"];
      lines.push(`    ${JSON.stringify(dk)}: ${open}`);
      items.forEach((it, m) => lines.push(`      ${it}${m < items.length - 1 ? "," : ""}`));
      lines.push(`    ${close}${c2}`);
    });
    lines.push(`  }${comma}`);
  });
  lines.push("}");
  return lines.join("\n");
}

function writeBox() {
  $("json").value = state.obj === null ? (state.sample?.raw ?? "") : formatJson(state.obj);
  state.jsonError = null;
}

function renderText() {
  renderQuestion($("text"), isObj(state.obj) ? state.obj.text : undefined);
  const gt = state.sample?.gt;
  const showGtText = state.showGt && isObj(gt);
  $("gt-text-wrap").hidden = !showGtText;
  if (showGtText) renderQuestion($("gt-text"), gt.text);
}

function badge(cls, text, title = "") {
  return `<span class="badge ${cls}"${title ? ` title="${esc(title)}"` : ""}>${esc(text)}</span>`;
}

// An invalid output can have dozens of near-identical errors; the rest fold away.
function listHtml(cls, items) {
  const li = (s) => `<li>${esc(s)}</li>`;
  let html = `<ul class="${cls}">${items.slice(0, MAX_LISTED).map(li).join("")}</ul>`;
  if (items.length > MAX_LISTED) {
    const rest = items.slice(MAX_LISTED);
    html += `<details><summary>${rest.length} more</summary><ul class="${cls}">${rest.map(li).join("")}</ul></details>`;
  }
  return html;
}

function renderStatus() {
  const s = state.sample;
  if (!s) {
    $("status").innerHTML = "";
    return;
  }
  const edited = snapshot() !== state.original;
  const current = state.checkedFor === snapshot() ? state.check : null;
  const b = [];
  if (state.jsonError) b.push(badge("bad", "box is not valid JSON"), badge("info", "showing its last valid version"));
  else if (!current) b.push(badge("info", "checking…"));
  else if (current.stage === "ok") b.push(badge("ok", "schema-valid"));
  else if (current.stage === "invalid") b.push(badge("bad", "schema-invalid"));
  else b.push(badge("bad", "not JSON"));
  if (edited) b.push(badge("warn", "edited"));
  else if (s.how === "fenced") b.push(badge("info", "fenced output", "the model wrapped its JSON in a code fence"));
  if (s.truncated) b.push(badge("bad", "truncated", "generation hit max_new_tokens"));

  let html = `<div class="badges">${b.join("")}</div>`;
  if (state.jsonError) html += listHtml("errors", [state.jsonError]);
  else if (current?.errors.length) html += listHtml("errors", current.errors);
  if (!state.jsonError && current?.warnings.length) html += listHtml("warnings", current.warnings);
  $("status").innerHTML = html;

  $("undo").disabled = state.history.length === 0;
  $("reset").disabled = !edited;
  $("render").disabled = current?.stage !== "ok";
}

function fmt(x, digits) {
  return x === null || x === undefined ? "–" : x.toFixed(digits);
}

function renderSampleInfo() {
  const s = state.sample;
  $("sample-id").textContent = s.id;
  const bits = [s.family, s.kind, s.n_tokens !== null ? `${s.n_tokens} tokens` : null].filter(Boolean);
  const summary = state.samples.find((x) => x.id === s.id);
  if (s.eval && summary) {
    const text = summary.text_exact === null ? "–" : summary.text_exact ? "exact" : "differs";
    bits.push(`text ${text}`, `structure F1 ${fmt(summary.structure_f1, 3)}`, `point error ${fmt(summary.point_error, 2)}`);
  }
  $("sample-meta").textContent = bits.join(" · ");
  $("raw").textContent = s.raw ?? "(no output)";
  const img = $("image");
  img.hidden = !s.has_image;
  if (s.has_image) img.src = `/api/image?${q({ run: state.run, id: s.id })}`;
  else img.removeAttribute("src");
}

// -- validation -------------------------------------------------------------------------------

let checkSeq = 0;
async function runCheck() {
  const seq = ++checkSeq;
  const snap = snapshot();
  if (state.obj === null) {
    // the raw output, unparsed and unedited: the server's verdict on it still holds
    state.check = state.sample.check;
    state.checkedFor = snap;
    renderStatus();
    return;
  }
  try {
    const res = await (await api("/api/check", { text: snap })).json();
    if (seq !== checkSeq) return; // a newer edit is being checked
    state.check = res;
    state.checkedFor = snap;
  } catch (e) {
    state.jsonError = `check failed: ${e.message}`;
  }
  renderStatus();
}
const runCheckSoon = debounce(runCheck, 150);

// After any change to state.obj: refresh what depends on it.
function afterEdit({ keepBox = false } = {}) {
  if (!keepBox) writeBox();
  renderText();
  renderStatus();
  redraw();
  runCheckSoon();
}

function setObj(obj, { keepBox = false } = {}) {
  state.history.push(snapshot());
  state.obj = obj;
  afterEdit({ keepBox });
}

// -- loading ----------------------------------------------------------------------------------

function updateHash() {
  history.replaceState(null, "", `#${q({ run: state.run, id: state.sid ?? "" })}`);
}

async function loadSample(id) {
  state.sid = id;
  highlightList();
  updateHash();
  const s = await getJson(`/api/sample?${q({ run: state.run, id })}`);
  if (state.sid !== id) return; // another sample was picked meanwhile
  state.sample = s;
  state.original = JSON.stringify(s.pred);
  state.obj = s.pred === null ? null : structuredClone(s.pred);
  state.history = [];
  state.check = s.check;
  state.checkedFor = state.original;
  state.drag = null;
  writeBox();
  renderSampleInfo();
  renderText();
  renderStatus();
  redraw();
}

const FILTERS = {
  all: () => true,
  diagram: (s) => s.pred === "diagram",
  invalid: (s) => s.stage !== "ok",
  errors: (s) => s.text_exact === false || (s.structure_f1 !== null && s.structure_f1 < 1),
};
const SORTS = {
  index: (a, b) => a.index - b.index,
  f1: (a, b) => (a.structure_f1 ?? 2) - (b.structure_f1 ?? 2) || a.index - b.index,
  err: (a, b) => (b.point_error ?? -1) - (a.point_error ?? -1) || a.index - b.index,
};

function metric(s) {
  if (s.structure_f1 !== null) return s.structure_f1.toFixed(2);
  if (s.pred === "null") return "no fig";
  return "";
}

function renderList() {
  state.visible = state.samples.filter(FILTERS[$("filter").value]).sort(SORTS[$("sort").value]);
  $("samples").innerHTML = state.visible
    .map(
      (s) =>
        `<button class="item" data-id="${esc(s.id)}" title="${esc(`${s.id}: ${s.stage}, predicted ${s.pred}`)}">` +
        `<span class="sdot ${s.stage}"></span><span class="sid">${esc(s.id)}</span>` +
        `<span class="metric">${metric(s)}</span></button>`,
    )
    .join("");
  $("count").textContent = `${state.visible.length} of ${state.samples.length}`;
  highlightList();
}

function highlightList() {
  for (const el of $("samples").querySelectorAll(".item")) {
    const on = el.dataset.id === state.sid;
    el.classList.toggle("current", on);
    if (on) el.scrollIntoView({ block: "nearest" });
  }
}

async function loadRun(key, id) {
  state.run = key;
  $("run").value = key;
  const info = state.runs.find((r) => r.key === key);
  $("filter").querySelector('[value="errors"]').disabled = !info?.has_eval;
  if (!info?.has_eval && $("filter").value === "errors") $("filter").value = "all";
  state.samples = await getJson(`/api/samples?${q({ run: key })}`);
  renderList();
  const pick = state.samples.some((s) => s.id === id) ? id : state.visible[0]?.id;
  if (pick) await loadSample(pick);
}

function step(delta) {
  if (!state.visible.length) return;
  const i = state.visible.findIndex((s) => s.id === state.sid);
  const j = i < 0 ? 0 : clamp(i + delta, 0, state.visible.length - 1);
  if (state.visible[j].id !== state.sid) loadSample(state.visible[j].id);
}

// -- dragging ---------------------------------------------------------------------------------

function toGrid(ev) {
  const r = $("svg").getBoundingClientRect();
  return [G.lo + ((ev.clientX - r.left) / r.width) * G.span, G.lo + ((ev.clientY - r.top) / r.height) * G.span];
}

function readout(text) {
  $("readout").textContent = text;
}

function onPointerDown(ev) {
  const d = diagramOf(state.obj);
  if (ev.button !== 0 || !d) return;
  const t = ev.target;
  const g = toGrid(ev);
  if (t.dataset.point !== undefined) {
    const p = d.points[t.dataset.point];
    // keep the grab offset, so the point does not jump to the pointer
    state.drag = { point: t.dataset.point, off: [p[0] - g[0], p[1] - g[1]], before: snapshot() };
  } else if (t.dataset.circle !== undefined) {
    state.drag = { circle: Number(t.dataset.circle), before: snapshot() };
  } else {
    return;
  }
  $("svg").setPointerCapture(ev.pointerId);
  ev.preventDefault();
  redraw();
}

function onPointerMove(ev) {
  const g = toGrid(ev);
  const dr = state.drag;
  if (!dr) {
    readout(`(${Math.round(g[0])}, ${Math.round(g[1])})`);
    return;
  }
  const d = state.obj.diagram;
  if (dr.point !== undefined) {
    const x = clamp(Math.round(g[0] + dr.off[0]), G.gmin, G.gmax);
    const y = clamp(Math.round(g[1] + dr.off[1]), G.gmin, G.gmax);
    const p = d.points[dr.point];
    if (p[0] !== x || p[1] !== y) {
      d.points[dr.point] = [x, y];
      redraw();
    }
    readout(`${dr.point} (${x}, ${y})`);
  } else {
    const c = d.circles[dr.circle];
    // one decimal, like the generator's radii
    const r = clamp(Math.round(dist(d.points[c.center], g) * 10) / 10, 0.1, G.gmax - G.gmin);
    if (r !== c.radius) {
      c.radius = r;
      redraw();
    }
    readout(`circle ${c.center}: radius ${r}`);
  }
}

function onPointerUp() {
  const dr = state.drag;
  if (!dr) return;
  state.drag = null;
  if (snapshot() !== dr.before) {
    state.history.push(dr.before);
    afterEdit();
  } else {
    redraw();
  }
}

// -- actions ----------------------------------------------------------------------------------

function undo() {
  if (!state.history.length) return;
  state.obj = JSON.parse(state.history.pop());
  afterEdit();
}

function reset() {
  if (snapshot() === state.original) return;
  setObj(JSON.parse(state.original));
}

// The valid output in canonical order (schema.output_to_obj), else the edit as it stands.
function exported() {
  const current = state.checkedFor === snapshot() ? state.check : null;
  return current?.stage === "ok" ? current.canonical : state.obj;
}

function download() {
  const blob = new Blob([`${JSON.stringify(exported(), null, 2)}\n`], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${state.sid.replace(/[^\w.-]+/g, "_")}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function copy() {
  await navigator.clipboard.writeText(JSON.stringify(exported()));
  readout("Copied the output JSON.");
}

let mplUrl = null;
async function renderMpl() {
  try {
    const res = await api("/api/render", { text: snapshot() });
    if (mplUrl) URL.revokeObjectURL(mplUrl);
    mplUrl = URL.createObjectURL(await res.blob());
    $("mpl-img").src = mplUrl;
    $("mpl").showModal();
  } catch (e) {
    readout(`render failed: ${e.message}`);
  }
}

function onBoxInput() {
  const text = $("json").value;
  let obj;
  try {
    obj = JSON.parse(text);
  } catch (e) {
    state.jsonError = e.message;
    renderStatus();
    return;
  }
  state.jsonError = null;
  if (JSON.stringify(obj) === snapshot()) {
    renderStatus();
    return;
  }
  setObj(obj, { keepBox: true });
}

function onKey(ev) {
  const typing = ev.target.closest?.("textarea, input, select");
  if ((ev.ctrlKey || ev.metaKey) && ev.key === "z" && !typing) {
    ev.preventDefault();
    undo();
    return;
  }
  if (typing || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.key === "ArrowRight" || ev.key === "j") step(1);
  else if (ev.key === "ArrowLeft" || ev.key === "k") step(-1);
}

// -- start ------------------------------------------------------------------------------------

async function init() {
  G = geometry(await getJson("/api/meta"));
  buildGrid();

  const svg = $("svg");
  svg.addEventListener("pointerdown", onPointerDown);
  svg.addEventListener("pointermove", onPointerMove);
  svg.addEventListener("pointerup", onPointerUp);
  svg.addEventListener("pointercancel", onPointerUp);
  svg.addEventListener("pointerleave", () => state.drag || readout(""));
  $("json").addEventListener("input", debounce(onBoxInput, EDIT_DEBOUNCE_MS));
  $("undo").addEventListener("click", undo);
  $("reset").addEventListener("click", reset);
  $("download").addEventListener("click", download);
  $("copy").addEventListener("click", copy);
  $("render").addEventListener("click", renderMpl);
  $("show-gt").addEventListener("change", (ev) => {
    state.showGt = ev.target.checked;
    renderText();
    redraw();
  });
  $("run").addEventListener("change", (ev) => loadRun(ev.target.value, null));
  for (const id of ["filter", "sort"]) $(id).addEventListener("change", renderList);
  $("samples").addEventListener("click", (ev) => {
    const item = ev.target.closest(".item");
    if (item) loadSample(item.dataset.id);
  });
  document.addEventListener("keydown", onKey);
  new ResizeObserver(redraw).observe($("stage"));

  state.runs = await getJson("/api/runs");
  if (!state.runs.length) {
    $("empty").hidden = false;
    $("empty").textContent = "No predictions under outputs/infer/. Run mathvlm.infer first.";
    return;
  }
  $("run").innerHTML = state.runs
    .map((r) => `<option value="${esc(r.key)}">${esc(r.key)} (${r.n}${r.has_eval ? ", eval" : ""})</option>`)
    .join("");
  const hash = new URLSearchParams(location.hash.slice(1));
  const fromHash = state.runs.find((r) => r.key === hash.get("run"));
  const first = fromHash ?? state.runs.find((r) => r.has_eval) ?? state.runs[0];
  await loadRun(first.key, hash.get("id"));
  window.addEventListener("hashchange", followHash);
}

// A pasted link or the back button: the page itself only replaces the hash, which does not
// fire hashchange.
function followHash() {
  const hash = new URLSearchParams(location.hash.slice(1));
  const run = hash.get("run");
  const id = hash.get("id");
  if (run !== state.run && state.runs.some((r) => r.key === run)) loadRun(run, id);
  else if (id && id !== state.sid && state.samples.some((s) => s.id === id)) loadSample(id);
}

init().catch((e) => {
  $("empty").hidden = false;
  $("empty").textContent = `Could not load the viewer: ${e.message}`;
});
