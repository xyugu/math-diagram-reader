// Browser port of render.py's _Painter: the same drawing conventions (listed in render.py's
// module docstring) and the same placement rules, so a diagram looks here the way it looks in
// the training images. Text is KaTeX instead of mathtext, so label boxes differ a little.
// Geometry is in grid units with y pointing down, which is also the SVG's user space.

const TAU = 2 * Math.PI;
const mod = (a, n) => ((a % n) + n) % n; // Python's %, which the ported formulas rely on

function unit(dx, dy) {
  const n = Math.hypot(dx, dy);
  return n < 1e-9 ? null : [dx / n, dy / n];
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const ang = (v) => Math.atan2(v[1], v[0]);
const midpoint = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
export const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function angdist(a, b) {
  const d = Math.abs(a - b) % TAU;
  return Math.min(d, TAU - d);
}

function distPointSegment(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  if (L2 === 0) return dist(p, a);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2));
  return dist(p, [a[0] + t * dx, a[1] + t * dy]);
}

// schema.canonical_segment / canonical_angle, as strings
const segKey = (s) => [s[0], s[1]].sort().join(" ");
const angleKey = (a) => `${a[1]} ${[a[0], a[2]].sort().join(" ")}`;

export const isObj = (x) => x !== null && typeof x === "object" && !Array.isArray(x);

const LABEL_KEYS = ["on", "angle", "arc", "at"];
function labelKind(lb) {
  const present = LABEL_KEYS.filter((k) => k in lb);
  return present.length === 1 ? present[0] : null;
}

// Keep only what can be drawn. A prediction (or a hand edit) may be schema-invalid: the page
// lists the validator's errors and still draws every well-formed element.
function sanitize(d) {
  const P = Object.create(null);
  if (isObj(d) && isObj(d.points)) {
    for (const [k, v] of Object.entries(d.points)) {
      if (Array.isArray(v) && v.length === 2 && v.every((c) => Number.isFinite(c))) {
        P[k] = [v[0], v[1]];
      }
    }
  }
  const list = (key) => (isObj(d) && Array.isArray(d[key]) ? d[key] : []);
  const known = (n) => typeof n === "string" && Object.hasOwn(P, n);
  const seg = (s) => Array.isArray(s) && s.length === 2 && known(s[0]) && known(s[1]) && s[0] !== s[1];
  const angle = (a) => Array.isArray(a) && a.length === 3 && a.every(known) && new Set(a).size === 3;
  const mark = (m) =>
    isObj(m) &&
    ((m.type === "right_angle" && Array.isArray(m.rays) && angle([m.rays[0], m.at, m.rays[1]])) ||
      ((m.type === "equal_length" || m.type === "parallel") &&
        Array.isArray(m.segments) &&
        m.segments.every(seg)) ||
      (m.type === "equal_angle" && Array.isArray(m.angles) && m.angles.every(angle)) ||
      (m.type === "arrow" && known(m.at) && known(m.from) && m.at !== m.from));
  const label = (lb) => {
    if (!isObj(lb) || typeof lb.text !== "string" || lb.text === "") return false;
    const kind = labelKind(lb);
    if (kind === "on") return seg(lb.on);
    if (kind === "angle") return angle(lb.angle);
    if (kind === "arc") return seg(lb.arc);
    return kind === "at" && Array.isArray(lb.at) && lb.at.length === 2 && lb.at.every((c) => Number.isFinite(c));
  };
  const circles = [];
  list("circles").forEach((c, index) => {
    if (isObj(c) && known(c.center) && Number.isFinite(c.radius) && c.radius > 0) {
      circles.push({ center: c.center, radius: c.radius, index }); // index: into d.circles
    }
  });
  return {
    P,
    segments: list("segments").filter(seg),
    circles,
    marks: list("marks").filter(mark),
    labels: list("labels").filter(label),
  };
}

const num = (v) => +v.toFixed(3);
const xy = (p) => `${num(p[0])},${num(p[1])}`;

// st: {ms, lw, markLw, dotR, labelDist, tol, pointDots, angleArcs, unnamed (RegExp),
//      measure(tex) -> [half width, half height] in grid units, background box included}
// Returns {svg, texts: [{tex, x, y, box}], P, circles}: SVG markup, the labels to place over
// it (centered on x, y), and the drawable points and circles (for drag handles).
export function paint(diagram, st) {
  const d = sanitize(diagram);
  const painter = new Painter(d, st);
  painter.draw();
  return { svg: painter.svg.join(""), texts: painter.texts, P: d.P, circles: d.circles };
}

class Painter {
  constructor(d, st) {
    this.d = d;
    this.st = st;
    this.P = d.P;
    const pts = Object.values(this.P);
    this.centroid = pts.length
      ? [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length]
      : [0, 0];
    this.arcRadius = new Map(); // angle key -> outermost arc radius drawn there
    this.obstacles = Object.create(null); // point -> directions its label should avoid
    for (const k of Object.keys(this.P)) this.obstacles[k] = [];
    this.arrowTips = new Set(d.marks.filter((m) => m.type === "arrow").map((m) => m.at));
    this.svg = [];
    this.texts = [];
  }

  draw() {
    this.drawCircles();
    this.drawSegments();
    this.drawArrows();
    this.drawRightAngles();
    this.drawSegmentDecorations();
    this.drawEqualAngles();
    this.drawAngleLabels();
    this.drawSegmentLabels();
    this.drawArcLabels();
    this.drawFreeText();
    this.drawPoints();
  }

  // -- primitives ---------------------------------------------------------------------------

  line(pts, w = this.st.lw) {
    this.svg.push(`<polyline points="${pts.map(xy).join(" ")}" stroke-width="${num(w)}"/>`);
  }

  markLine(pts) {
    this.line(pts, this.st.markLw);
  }

  arc(v, a1, a2, r) {
    const delta = mod(a2 - a1 + Math.PI, TAU) - Math.PI; // shorter way round
    const pts = [];
    for (let i = 0; i <= 32; i++) {
      const t = a1 + (delta * i) / 32;
      pts.push([v[0] + r * Math.cos(t), v[1] + r * Math.sin(t)]);
    }
    this.markLine(pts);
  }

  text(x, y, tex) {
    this.texts.push({ tex, x, y, box: true });
  }

  // -- passes -------------------------------------------------------------------------------

  drawCircles() {
    for (const c of this.d.circles) {
      const [cx, cy] = this.P[c.center];
      this.svg.push(
        `<circle cx="${num(cx)}" cy="${num(cy)}" r="${num(c.radius)}" stroke-width="${num(this.st.lw)}"/>`,
      );
    }
  }

  drawSegments() {
    for (const [a, b] of this.d.segments) this.line([this.P[a], this.P[b]]);
  }

  arrowhead(m) {
    const tip = this.P[m.at];
    const u = unit(...sub(tip, this.P[m.from]));
    if (!u) return null;
    const length = this.st.ms;
    const half = 0.42 * this.st.ms;
    const base = [tip[0] - u[0] * length, tip[1] - u[1] * length];
    const n = [-u[1], u[0]];
    return [tip, [base[0] + n[0] * half, base[1] + n[1] * half], [base[0] - n[0] * half, base[1] - n[1] * half]];
  }

  drawArrows() {
    for (const m of this.d.marks) {
      if (m.type !== "arrow") continue;
      const tri = this.arrowhead(m);
      if (!tri) continue;
      this.svg.push(`<polygon points="${tri.map(xy).join(" ")}"/>`);
      this.obstacles[m.at].push(ang(unit(...sub(this.P[m.from], this.P[m.at]))));
    }
  }

  bisector(v, e1, e2) {
    const u1 = unit(...sub(e1, v));
    const u2 = unit(...sub(e2, v));
    if (!u1 || !u2) return null;
    let b = unit(u1[0] + u2[0], u1[1] + u2[1]);
    if (!b) {
      // straight angle: the normal pointing away from the centroid
      const n = [-u1[1], u1[0]];
      const away = sub(v, this.centroid);
      b = n[0] * away[0] + n[1] * away[1] >= 0 ? n : [-n[0], -n[1]];
    }
    return b;
  }

  drawRightAngles() {
    for (const m of this.d.marks) {
      if (m.type !== "right_angle") continue;
      const v = this.P[m.at];
      const e1 = this.P[m.rays[0]];
      const e2 = this.P[m.rays[1]];
      const u1 = unit(...sub(e1, v));
      const u2 = unit(...sub(e2, v));
      if (!u1 || !u2) continue;
      const side = Math.min(this.st.ms, 0.35 * dist(v, e1), 0.35 * dist(v, e2));
      const p1 = [v[0] + side * u1[0], v[1] + side * u1[1]];
      const p2 = [p1[0] + side * u2[0], p1[1] + side * u2[1]];
      const p3 = [v[0] + side * u2[0], v[1] + side * u2[1]];
      this.markLine([p1, p2, p3]);
      const b = this.bisector(v, e1, e2);
      if (b) this.obstacles[m.at].push(ang(b));
    }
  }

  segmentDecorations() {
    const decos = new Map(); // segment key -> {seg, items: [{kind, count, dir}]}
    const add = (s, item) => {
      const k = segKey(s);
      if (!decos.has(k)) decos.set(k, { seg: s, items: [] });
      decos.get(k).items.push(item);
    };
    let nEq = 0;
    let nPar = 0;
    for (const m of this.d.marks) {
      if (m.type === "equal_length") {
        nEq += 1;
        for (const s of m.segments) {
          const u = unit(...sub(this.P[s[1]], this.P[s[0]]));
          if (u) add(s, { kind: "ticks", count: Math.min(nEq, 4), dir: u });
        }
      } else if (m.type === "parallel") {
        nPar += 1;
        let ref = null;
        for (const s of m.segments) {
          let u = unit(...sub(this.P[s[1]], this.P[s[0]]));
          if (!u) continue;
          if (!ref) ref = u;
          else if (u[0] * ref[0] + u[1] * ref[1] < 0) u = [-u[0], -u[1]]; // same way in a group
          add(s, { kind: "arrows", count: Math.min(nPar, 4), dir: u });
        }
      }
    }
    return decos;
  }

  drawSegmentDecorations() {
    const ms = this.st.ms;
    const spacing = 0.45 * ms;
    const half = 0.55 * ms;
    const gap = 0.7 * ms;
    for (const {
      seg: [a, b],
      items,
    } of this.segmentDecorations().values()) {
      const pa = this.P[a];
      const pb = this.P[b];
      const mid = midpoint(pa, pb);
      const widths = items.map((it) => (it.count - 1) * spacing + (it.kind === "arrows" ? half : 0));
      const total = widths.reduce((s, w) => s + w, 0) + gap * (items.length - 1);
      const base = unit(...sub(pb, pa));
      if (!base) continue;
      let cursor = -total / 2;
      items.forEach((it, j) => {
        const d = it.dir;
        const n = [-d[1], d[0]];
        for (let i = 0; i < it.count; i++) {
          const t = cursor + i * spacing;
          const c = [mid[0] + base[0] * t, mid[1] + base[1] * t];
          if (it.kind === "ticks") {
            this.markLine([
              [c[0] - n[0] * half, c[1] - n[1] * half],
              [c[0] + n[0] * half, c[1] + n[1] * half],
            ]);
          } else {
            const tip = [c[0] + d[0] * half * 0.5, c[1] + d[1] * half * 0.5];
            const back = [tip[0] - d[0] * half, tip[1] - d[1] * half];
            this.markLine([
              [back[0] + n[0] * half * 0.7, back[1] + n[1] * half * 0.7],
              tip,
              [back[0] - n[0] * half * 0.7, back[1] - n[1] * half * 0.7],
            ]);
          }
        }
        cursor += widths[j] + gap;
      });
    }
  }

  // Arc radius that grows for narrow angles, so the arc stays long enough to read (and to
  // carry ticks), but never reaches past 45% of the shorter ray.
  arcRadiusFor(v, e1, e2, minArcLen) {
    const u1 = unit(...sub(e1, v));
    const u2 = unit(...sub(e2, v));
    const angle = u1 && u2 ? angdist(ang(u1), ang(u2)) : Math.PI;
    const r = Math.max(1.3 * this.st.ms, minArcLen / Math.max(angle, 1e-3));
    const cap = 0.45 * Math.min(dist(v, e1), dist(v, e2));
    return Math.max(Math.min(r, cap), 0.5 * this.st.ms);
  }

  drawEqualAngles() {
    const ms = this.st.ms;
    let k = 0;
    for (const m of this.d.marks) {
      if (m.type !== "equal_angle") continue;
      k += 1;
      for (const a of m.angles) {
        const v = this.P[a[1]];
        const e1 = this.P[a[0]];
        const e2 = this.P[a[2]];
        const u1 = unit(...sub(e1, v));
        const u2 = unit(...sub(e2, v));
        if (!u1 || !u2) continue;
        const r = this.arcRadiusFor(v, e1, e2, 1.8 * ms);
        const t1 = ang(u1);
        const t2 = ang(u2);
        this.arc(v, t1, t2, r);
        // k short radial ticks across the arc's middle
        const mid = t1 + (mod(t2 - t1 + Math.PI, TAU) - Math.PI) / 2;
        const h = 0.35 * ms;
        const step = (0.45 * ms) / r;
        const nTicks = Math.min(k, 4);
        for (let j = 0; j < nTicks; j++) {
          const t = mid + (j - (nTicks - 1) / 2) * step;
          const c = Math.cos(t);
          const s = Math.sin(t);
          this.markLine([
            [v[0] + (r - h) * c, v[1] + (r - h) * s],
            [v[0] + (r + h) * c, v[1] + (r + h) * s],
          ]);
        }
        this.arcRadius.set(angleKey(a), r + h);
        const b = this.bisector(v, e1, e2);
        if (b) this.obstacles[a[1]].push(ang(b));
      }
    }
  }

  drawAngleLabels() {
    const margin = 0.35 * this.st.ms;
    for (const lb of this.d.labels) {
      if (labelKind(lb) !== "angle") continue;
      const a = lb.angle;
      const v = this.P[a[1]];
      const e1 = this.P[a[0]];
      const e2 = this.P[a[2]];
      const b = this.bisector(v, e1, e2);
      if (!b) {
        // coincident points: just above the vertex
        this.text(v[0], v[1] - 2 * this.st.ms, lb.text);
        continue;
      }
      const u1 = unit(...sub(e1, v));
      const u2 = unit(...sub(e2, v));
      let rArc = this.arcRadius.get(angleKey(a));
      if (rArc === undefined && this.st.angleArcs) {
        rArc = this.arcRadiusFor(v, e1, e2, 1.2 * this.st.ms);
        this.arc(v, ang(u1), ang(u2), rArc);
      }
      rArc = rArc || 0;
      // distance along the bisector at which the label clears the arc and both sides
      const [hw, hh] = this.st.measure(lb.text);
      let r = rArc + margin + hw * Math.abs(b[0]) + hh * Math.abs(b[1]);
      for (const u of [u1, u2]) {
        let n = [-u[1], u[0]];
        let nb = n[0] * b[0] + n[1] * b[1];
        if (nb < 0) {
          n = [-n[0], -n[1]];
          nb = -nb;
        }
        if (nb > 1e-3) r = Math.max(r, (margin + hw * Math.abs(n[0]) + hh * Math.abs(n[1])) / nb);
      }
      r = Math.min(r, Math.max(0.9 * Math.min(dist(v, e1), dist(v, e2)), rArc + margin));
      this.text(v[0] + b[0] * r, v[1] + b[1] * r, lb.text);
      this.obstacles[a[1]].push(ang(b));
    }
  }

  drawSegmentLabels() {
    const margin = 0.3 * this.st.ms;
    const decos = this.segmentDecorations();
    // a length label must not sit on points or on the marks at decorated midpoints
    const decoCenters = [...decos.values()].map(({ seg: [a, b] }) => midpoint(this.P[a], this.P[b]));
    const clear = 2.5 * this.st.ms;
    for (const lb of this.d.labels) {
      if (labelKind(lb) !== "on") continue;
      const pa = this.P[lb.on[0]];
      const pb = this.P[lb.on[1]];
      const own = decos.has(segKey(lb.on));
      const ownMid = midpoint(pa, pb);
      const others = Object.entries(this.P)
        .filter(([n]) => n !== lb.on[0] && n !== lb.on[1])
        .map(([, p]) => p);
      const obstacles = others.concat(
        decoCenters.filter((c) => !own || c[0] !== ownMid[0] || c[1] !== ownMid[1]),
      );
      let mid = ownMid;
      for (const t of [0.5, 0.35, 0.65, 0.25, 0.75]) {
        // slide along the segment if crowded
        const cand = [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t];
        if (obstacles.every((o) => dist(cand, o) >= clear)) {
          mid = cand;
          break;
        }
      }
      const u = unit(...sub(pb, pa));
      let n = u ? [-u[1], u[0]] : [0, -1];
      const away = sub(mid, this.centroid);
      const dot = n[0] * away[0] + n[1] * away[1];
      if (dot < -1e-6 || (Math.abs(dot) <= 1e-6 && n[1] > 0)) n = [-n[0], -n[1]];
      const [hw, hh] = this.st.measure(lb.text);
      let off = margin + hw * Math.abs(n[0]) + hh * Math.abs(n[1]);
      if (own) off += 0.55 * this.st.ms; // clear the ticks/arrows at the midpoint
      this.text(mid[0] + n[0] * off, mid[1] + n[1] * off, lb.text);
    }
  }

  // Outside the circle through both endpoints, on the radius through the minor arc's
  // midpoint; without such a circle, beside the chord's midpoint, away from the figure.
  arcLabelAnchor(lb) {
    const a = this.P[lb.arc[0]];
    const b = this.P[lb.arc[1]];
    const [hw, hh] = this.st.measure(lb.text);
    const margin = 0.3 * this.st.ms;
    let best = null;
    for (const c of this.d.circles) {
      const o = this.P[c.center];
      const err = Math.abs(dist(a, o) - c.radius) + Math.abs(dist(b, o) - c.radius);
      if (!best || err < best.err) best = { err, o, r: c.radius };
    }
    let o;
    let r;
    let u;
    if (best && best.err <= 2 * (this.st.tol + 1)) {
      ({ o, r } = best);
      const t1 = ang(sub(a, o));
      const t2 = ang(sub(b, o));
      const tm = t1 + (mod(t2 - t1 + Math.PI, TAU) - Math.PI) / 2;
      u = [Math.cos(tm), Math.sin(tm)];
    } else {
      o = midpoint(a, b);
      r = 0;
      u = unit(...sub(o, this.centroid)) || [0, -1];
    }
    const off = r + margin + hw * Math.abs(u[0]) + hh * Math.abs(u[1]);
    return [o[0] + u[0] * off, o[1] + u[1] * off];
  }

  drawArcLabels() {
    for (const lb of this.d.labels) {
      if (labelKind(lb) === "arc") this.text(...this.arcLabelAnchor(lb), lb.text);
    }
  }

  drawFreeText() {
    for (const lb of this.d.labels) {
      if (labelKind(lb) === "at") this.text(lb.at[0], lb.at[1], lb.text);
    }
  }

  rayDirections(name) {
    const p = this.P[name];
    const dirs = [];
    for (const [a, b] of this.d.segments) {
      if (name === a || name === b) {
        const u = unit(...sub(this.P[a === name ? b : a], p));
        if (u) dirs.push(ang(u));
      } else if (distPointSegment(p, this.P[a], this.P[b]) <= this.st.tol) {
        for (const end of [a, b]) {
          // a point inside a segment: two rays
          const u = unit(...sub(this.P[end], p));
          if (u) dirs.push(ang(u));
        }
      }
    }
    for (const c of this.d.circles) {
      const radial = unit(...sub(p, this.P[c.center]));
      if (radial && Math.abs(dist(p, this.P[c.center]) - c.radius) <= this.st.tol) {
        const t = ang(radial);
        dirs.push(t + Math.PI / 2, t - Math.PI / 2); // tangent directions
      }
    }
    return dirs;
  }

  // The candidate direction with the most angular clearance from drawn rays and marks, with a
  // mild preference for pointing away from the figure's centroid.
  labelDirection(name) {
    const p = this.P[name];
    const avoid = this.rayDirections(name).concat(this.obstacles[name]);
    const away = unit(...sub(p, this.centroid));
    const awayT = away ? ang(away) : -Math.PI / 2;
    let best = awayT;
    let bestScore = -Infinity;
    for (let i = 0; i < 24; i++) {
      const t = -Math.PI + i * (TAU / 24);
      const clearance = avoid.length ? Math.min(...avoid.map((r) => angdist(t, r))) : Math.PI;
      const score = clearance + 0.5 * Math.cos(t - awayT);
      if (score > bestScore + 1e-9) {
        best = t;
        bestScore = score;
      }
    }
    return best;
  }

  drawPoints() {
    const centers = new Set(this.d.circles.map((c) => c.center));
    for (const [name, p] of Object.entries(this.P)) {
      if ((this.st.pointDots || centers.has(name)) && !this.arrowTips.has(name)) {
        this.svg.push(`<circle class="dot" cx="${num(p[0])}" cy="${num(p[1])}" r="${num(this.st.dotR)}"/>`);
      }
      if (this.st.unnamed.test(name)) continue;
      const t = this.labelDirection(name);
      const r = this.st.labelDist;
      this.texts.push({ tex: name, x: p[0] + r * Math.cos(t), y: p[1] + r * Math.sin(t), box: false });
    }
  }
}
