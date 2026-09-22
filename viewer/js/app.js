// Atlas viewer.
//
// Reading order: binary.js (decode) -> hierarchy.js (LOD rule) ->
// labels.js (placement) -> scene.js (WebGL) -> this file (glue and input).

import { decodeKnn, decodeTile } from './binary.js';
import { Hierarchy, PRUNE_PURITY, VISIBLE_MAX, VISIBLE_MIN } from './hierarchy.js';
import { drawLabels } from './labels.js';
import { createScene } from './scene.js';
import { Grid } from './spatial.js';
import { installCapture } from './capture.js';

const $ = (id) => document.getElementById(id);

async function boot() {
  const status = $('status');
  status.textContent = 'loading…';

  const t_start = performance.now();
  const [tileBuf, manifest, meta, knnBuf] = await Promise.all([
    fetch('public/all.bin').then(r => r.arrayBuffer()),
    fetch('public/manifest.json').then(r => r.json()),
    fetch('public/meta.json').then(r => r.json()),
    fetch('public/knn.bin').then(r => r.arrayBuffer()),
  ]);
  const t_fetched = performance.now();

  const P = decodeTile(tileBuf);
  const KNN = decodeKnn(knnBuf);
  const H = new Hierarchy(manifest);
  const t_decoded = performance.now();

  const canvas = $('gl');
  const overlay = $('overlay');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const ctx = overlay.getContext('2d');

  function resize() {
    for (const el of [canvas, overlay]) {
      el.width = el.clientWidth * dpr;
      el.height = el.clientHeight * dpr;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  resize();

  // preserveDrawingBuffer lets the capture path read finished frames back out
  // of the canvas. It costs a little throughput and is the only way to make
  // the demo GIF the real renderer rather than a re-creation of it.
  const regl = createREGL({ canvas, attributes: {
    antialias: false, alpha: false, preserveDrawingBuffer: true } });
  const scene = createScene(regl, P, dpr);

  const [minx, miny, maxx, maxy] = P.bbox;

  // Fit the home view to where the points actually are, not to the extreme
  // ones. A handful of outliers sit far outside the body of the map, so
  // fitting the raw bounding box leaves the catalog as a small island in a
  // large empty frame -- the same outlier problem that inflated the category
  // boxes, one level up.
  const coreExtent = (() => {
    const xs = Float32Array.from(P.x).sort();
    const ys = Float32Array.from(P.y).sort();
    const q = (a, p) => a[Math.floor((a.length - 1) * p)];
    return [q(xs, 0.005), q(ys, 0.005), q(xs, 0.995), q(ys, 0.995)];
  })();
  const [cx0, cy0, cx1, cy1] = coreExtent;
  const home = {
    cx: (cx0 + cx1) / 2,
    cy: (cy0 + cy1) / 2,
    scale: 2 / (Math.max(cx1 - cx0, cy1 - cy0) * 1.08),
  };
  const view = { ...home };

  const grid = new Grid(P.x, P.y, P.bbox);

  // ---- projection -------------------------------------------------------
  const aspect = () => canvas.width / canvas.height;
  function project(wx, wy) {
    const cx = (wx - view.cx) * view.scale / aspect();
    const cy = (wy - view.cy) * view.scale;
    return [(cx * 0.5 + 0.5) * overlay.width, (0.5 - cy * 0.5) * overlay.height];
  }
  function unproject(px, py) {
    const cx = (px / overlay.width * 2 - 1) * aspect();
    const cy = 1 - py / overlay.height * 2;
    return [cx / view.scale + view.cx, cy / view.scale + view.cy];
  }

  // ---- colour by active ancestor ----------------------------------------
  let colourKey = '';
  function refreshColours() {
    const key = view.scale.toFixed(4);
    if (key === colourKey) return;
    colourKey = key;
    const cache = new Map();
    scene.setColours((i) => {
      const leaf = P.cat[i];
      let idx = cache.get(leaf);
      if (idx === undefined) {
        // A territory keeps one hue while it is the active level. Once the
        // view is close enough that a descendant becomes active, siblings are
        // offset from the territory's base hue so the territory visibly splits
        // into coloured sub-regions -- without any point moving.
        const top = H.topAncestor(leaf);
        const base = H.topIndex.get(top) ?? 0;
        const active = H.activeAncestorOf(leaf, project, overlay.width, overlay.height);
        idx = (active === top)
          ? base
          : base + 1 + (H.byId.get(active)?.size ?? active) % 5;
        cache.set(leaf, idx);
      }
      return idx;
    });
  }

  // ---- focus / neighbour highlight --------------------------------------
  let focus = -1;
  function setFocus(i) {
    focus = i;
    if (i < 0) { scene.clearHighlight(); renderPanel(null); return; }
    const nb = Array.from(KNN.neighboursOf(i));
    scene.setHighlight(i, nb);
    renderPanel(i, nb);
  }

  function renderPanel(i, nb) {
    const panel = $('panel');
    if (i === null || i === undefined || i < 0) { panel.hidden = true; return; }
    panel.hidden = false;
    const leaf = H.byId.get(P.cat[i]);
    const path = H.chain(P.cat[i]).map(id => H.byId.get(id).label).reverse();

    // how scattered are this game's true neighbours? spread relative to the
    // whole map is the honest answer to "is the map telling the truth here"
    let sx = 0, sy = 0;
    for (const n of nb) { sx += P.x[n]; sy += P.y[n]; }
    sx /= nb.length; sy /= nb.length;
    let spread = 0;
    for (const n of nb) spread += Math.hypot(P.x[n] - sx, P.y[n] - sy);
    spread /= nb.length;
    const mapSpan = Math.max(maxx - minx, maxy - miny);
    const rel = spread / mapSpan;
    const verdict = rel < 0.05
      ? ['cohesive', 'its true neighbours sit together here']
      : rel < 0.14
        ? ['partly scattered', 'some neighbours are elsewhere on the map']
        : ['scattered', 'this game resists placement — its neighbours are spread across the map'];

    panel.innerHTML = `
      <div class="name">${escapeHtml(meta.names[i] || '—')}</div>
      <div class="tags">${(meta.tags[i] || []).map(escapeHtml).join(' · ')}</div>
      <div class="path">${path.map(escapeHtml).join(' <span>›</span> ')}</div>
      <div class="nb">
        <b>${nb.length}</b> nearest in the embedding —
        <span class="verdict ${verdict[0].split(' ')[0]}">${verdict[0]}</span>
        <div class="hint">${verdict[1]}</div>
        <ol>${nb.slice(0, 6).map(n => `<li>${escapeHtml(meta.names[n] || '—')}</li>`).join('')}</ol>
      </div>
      <div class="foot">${(meta.reviews[i] || 0).toLocaleString()} reviews ·
        cluster “${escapeHtml(leaf ? leaf.label : '')}” (${leaf ? leaf.confidence : '—'})</div>`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // ---- frame ------------------------------------------------------------
  let lastLabelStats = { drawn: 0, suppressed: 0, unlabelled: 0 };
  let frames = 0, fps = 0, lastT = performance.now();

  function drawFrame() {
    refreshColours();
    regl.poll();
    regl.clear({ color: [0.051, 0.067, 0.09, 1], depth: 1 });
    scene.draw({ view, dim: focus >= 0 ? 1 : 0 });

    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const entries = H.visibleLabels(project, overlay.width, overlay.height);
    lastLabelStats = drawLabels(ctx, entries, project, dpr);

    frames++;
    const now = performance.now();
    if (now - lastT > 500) {
      fps = Math.round(frames * 1000 / (now - lastT));
      frames = 0; lastT = now;
    }
    $('hud').innerHTML =
        `<b>${P.count.toLocaleString()}</b> <span>points</span>`
      + ` &nbsp;<b>${(view.scale / home.scale).toFixed(1)}×</b> <span>zoom</span>`
      + ` &nbsp;<b>${lastLabelStats.drawn}</b> <span>labels</span>`
      + (lastLabelStats.suppressed
          ? ` <span>(${lastLabelStats.suppressed} hidden)</span>` : '')
      + ` &nbsp;<b>${fps || '—'}</b> <span>fps</span>`;
  }

  function frame() { drawFrame(); requestAnimationFrame(frame); }

  // ---- input ------------------------------------------------------------
  let dragging = false, moved = false, lx = 0, ly = 0;
  canvas.addEventListener('mousedown', e => {
    dragging = true; moved = false; lx = e.clientX; ly = e.clientY;
  });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', e => {
    if (dragging) {
      moved = true;
      view.cx -= (e.clientX - lx) / canvas.clientWidth * 2 / view.scale * aspect();
      view.cy += (e.clientY - ly) / canvas.clientHeight * 2 / view.scale;
      lx = e.clientX; ly = e.clientY;
      return;
    }
    const r = canvas.getBoundingClientRect();
    const [wx, wy] = unproject((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
    const radius = 6 * dpr / view.scale / overlay.width * 2 * aspect();
    const hit = grid.nearest(wx, wy, radius * 2);
    if (hit !== focus) setFocus(hit);
  });
  canvas.addEventListener('mouseleave', () => setFocus(-1));
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const [wx, wy] = unproject((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
    const k = Math.exp(-e.deltaY * 0.0016);
    const next = Math.max(home.scale * 0.5, Math.min(home.scale * 4000, view.scale * k));
    // zoom about the cursor
    view.cx = wx - (wx - view.cx) * (view.scale / next);
    view.cy = wy - (wy - view.cy) * (view.scale / next);
    view.scale = next;
  }, { passive: false });
  window.addEventListener('resize', resize);
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape') { Object.assign(view, home); setFocus(-1); }
  });

  status.remove();
  requestAnimationFrame(frame);

  // ---- diagnostics for the benchmark harness ----------------------------
  const px = new Uint8Array(4);
  function benchFrames(n = 60) {
    const gl = regl._gl;
    scene.draw({ view, dim: 0 });
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const t0 = performance.now();
    for (let i = 0; i < n; i++) scene.draw({ view, dim: 0 });
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const ms = (performance.now() - t0) / n;
    return { msPerFrame: +ms.toFixed(3), impliedFps: Math.round(1000 / ms) };
  }

  installCapture(
    {
      view, home, P, setFocus,
      neighbours: (i) => Array.from(KNN.neighboursOf(i)),
      pathOf: (i) => H.chain(P.cat[i])
        .map(id => H.byId.get(id).label).reverse().join('  ›  '),
    },
    { canvas, overlay, meta, drawFrame });

  window.__atlas = {
    P, KNN, H, view, home, scene, regl, project, unproject, grid,
    benchFrames, setFocus,
    labelStats: () => lastLabelStats,
    timing: {
      fetch_ms: +(t_fetched - t_start).toFixed(1),
      decode_ms: +(t_decoded - t_fetched).toFixed(1),
      first_paint_ms: +(performance.now() - t_start).toFixed(1),
    },
    constants: { VISIBLE_MIN, VISIBLE_MAX, PRUNE_PURITY },
  };
  document.dispatchEvent(new CustomEvent('atlas-ready'));
}

boot().catch(err => {
  const s = document.getElementById('status');
  if (s) s.textContent = 'error: ' + err.message;
  console.error(err);
});
