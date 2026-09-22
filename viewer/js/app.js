// Atlas viewer.
//
// Reading order: binary.js (decode) -> tilestore.js (streaming) ->
// hierarchy.js (LOD rule) -> labels.js (placement) -> scene.js (WebGL) ->
// this file (glue and input).
//
// Cold load fetches the manifest and the tiles covering the opening viewport.
// Metadata and neighbour lists arrive on first hover.

import { decodeKnn } from './binary.js';
import { HoverCard, ImageCache, STORE_URL } from './card.js';
import { Hierarchy, PRUNE_PURITY, VISIBLE_MAX, VISIBLE_MIN } from './hierarchy.js';
import { drawLabels } from './labels.js';
import { MetaStore } from './meta.js';
import { colourFor, createScene } from './scene.js';
import { Grid } from './spatial.js';
import { TileStore, flatten } from './tilestore.js';
import { installCapture } from './capture.js';
import { SearchBox } from './searchbox.js';

const $ = (id) => document.getElementById(id);

async function boot() {
  const status = $('status');
  const t_start = performance.now();

  const manifest = await fetch('public/manifest.json').then(r => r.json());
  const H = new Hierarchy(manifest);
  const meta = new MetaStore({ shardSize: manifest.meta_shard });

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

  const regl = createREGL({ canvas, attributes: {
    antialias: false, alpha: false, preserveDrawingBuffer: true } });
  const scene = createScene(regl, dpr);

  // Fit to the data's core extent, not its raw bounds: a handful of outliers
  // sit far outside the body of the map, and fitting those leaves the catalog
  // a small island in a large empty frame. Fitting both axes to the viewport
  // also makes the points meaningfully larger on screen.
  const [cx0, cy0, cx1, cy1] = manifest.core_extent;
  const aspect = () => canvas.width / canvas.height;
  const fitScale = () => {
    const w = (cx1 - cx0) * 1.04, h = (cy1 - cy0) * 1.04;
    return Math.min(2 * aspect() / w, 2 / h);
  };
  const home = { cx: (cx0 + cx1) / 2, cy: (cy0 + cy1) / 2, scale: fitScale() };
  const view = { ...home };

  const images = new ImageCache();
  const card = new HoverCard($('card'), images);

  const tiles = new TileStore({
    bounds: manifest.bounds, maxZoom: manifest.max_zoom, capacity: 200,
  });

  // ---- projection -------------------------------------------------------
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

  // ---- streaming --------------------------------------------------------
  let visibleKey = '';
  let grid = null;
  let pending = null;

  async function refreshTiles(force = false) {
    const z = tiles.levelFor(view.scale, home.scale);
    const keys = tiles.tilesFor(z, view, aspect());
    const key = keys.join(',');
    if (key === visibleKey && !force) return;
    visibleKey = key;

    // draw whatever is already cached immediately; a zoom should never flash
    // empty while the network catches up
    const ready = tiles.cached(keys);
    if (ready.length) applyPoints(flatten(ready));

    pending = key;
    await tiles.ensure(keys);
    if (pending !== key) return;          // a newer viewport superseded this one
    const got = tiles.cached(keys);
    if (got.length) applyPoints(flatten(got));
  }

  function applyPoints(P) {
    if (!P.count) return;
    scene.setPoints(P);
    grid = new Grid(P.x, P.y, manifest.bounds, 96);
    colourKey = '';                        // force a recolour for the new set
  }

  // ---- colour by active ancestor ----------------------------------------
  let colourKey = '';
  function refreshColours() {
    const key = view.scale.toFixed(4) + ':' + scene.count;
    if (key === colourKey) return;
    colourKey = key;
    const cache = new Map();
    const P = scene.points;
    scene.setColours((i) => {
      const leaf = P.cat[i];
      let idx = cache.get(leaf);
      if (idx === undefined) {
        // A territory keeps one hue while it is the active level. Once a
        // descendant becomes active, siblings offset from the territory's base
        // hue so the territory visibly splits -- without any point moving.
        const top = H.topAncestor(leaf);
        const base = H.topIndex.get(top) ?? 0;
        const active = H.activeAncestorOf(leaf, project, overlay.width, overlay.height);
        idx = (active === top)
          ? base
          : base + 1 + ((H.byId.get(active)?.size ?? active) % 5);
        cache.set(leaf, idx);
      }
      return idx;
    });
  }

  // ---- focus / neighbour highlight --------------------------------------
  let KNN = null, knnLoading = null;
  function ensureKnn() {
    if (KNN) return Promise.resolve(KNN);
    if (!knnLoading) {
      knnLoading = fetch('public/knn.bin')
        .then(r => r.arrayBuffer())
        .then(b => { KNN = decodeKnn(b); return KNN; });
    }
    return knnLoading;
  }

  let focus = -1;
  let cursor = [0, 0];

  async function setFocus(gid, showCard = true) {
    focus = gid;
    if (gid < 0) { scene.clearHighlight(); renderPanel(null); card.hide(); return; }
    const knn = await ensureKnn();
    if (focus !== gid) return;
    const nb = Array.from(knn.neighboursOf(gid));
    scene.setHighlight(gid, nb);
    await meta.ensure([gid, ...nb]);
    if (focus !== gid) return;
    renderPanel(gid, nb);
    if (showCard) showCardFor(gid, nb);
  }

  function showCardFor(gid, nb) {
    const m = meta.get(gid);
    if (!m) return;
    const slot = scene.indexOfId(gid);
    const cat = slot === undefined ? null : H.byId.get(scene.points.cat[slot]);
    const colour = cat ? colourOfCategory(cat) : [0.5, 0.6, 0.7];

    // Screen positions of the haloed neighbours, so the card can flip away
    // from them rather than covering the thing it is meant to explain.
    const avoid = [];
    for (const n of nb) {
      const k = scene.indexOfId(n);
      if (k === undefined) continue;
      const [sx, sy] = project(scene.points.x[k], scene.points.y[k]);
      avoid.push([sx / dpr, sy / dpr]);
    }
    card.show(m, colour, cursor[0], cursor[1], avoid);
  }

  function renderPanel(gid, nb) {
    const panel = $('panel');
    if (gid === null || gid === undefined || gid < 0) { panel.hidden = true; return; }
    const m = meta.get(gid);
    const slot = scene.indexOfId(gid);
    if (!m || slot === undefined) { panel.hidden = true; return; }
    panel.hidden = false;

    const leaf = H.byId.get(scene.points.cat[slot]);
    const path = leaf
      ? H.chain(leaf.id).map(id => H.byId.get(id).label).reverse()
      : [];

    // Spread of the true neighbours, over those currently loaded. Neighbours
    // outside the viewport are not in the buffer, which is itself information:
    // it means they are far away.
    const px = scene.points;
    let sx = 0, sy = 0, k = 0;
    for (const n of nb) {
      const s = scene.indexOfId(n);
      if (s === undefined) continue;
      sx += px.x[s]; sy += px.y[s]; k++;
    }
    let verdict = ['scattered', 'most of its neighbours are outside this view'];
    if (k >= 3) {
      sx /= k; sy /= k;
      let spread = 0;
      for (const n of nb) {
        const s = scene.indexOfId(n);
        if (s === undefined) continue;
        spread += Math.hypot(px.x[s] - sx, px.y[s] - sy);
      }
      spread /= k;
      const span = Math.max(cx1 - cx0, cy1 - cy0);
      const rel = spread / span;
      verdict = rel < 0.05
        ? ['cohesive', 'its true neighbours sit together here']
        : rel < 0.14
          ? ['partly', 'some neighbours are elsewhere on the map']
          : ['scattered', 'this game resists placement — its neighbours are spread out'];
    }

    const names = nb.slice(0, 6).map(n => meta.get(n)?.name || '…');
    panel.innerHTML = `
      <div class="name">${esc(m.name)}</div>
      <div class="tags">${m.tags.map(esc).join(' · ')}</div>
      <div class="path">${path.map(esc).join(' <span>›</span> ')}</div>
      <div class="nb">
        <b>${nb.length}</b> nearest in the embedding —
        <span class="verdict ${verdict[0]}">${verdict[0]}</span>
        <div class="hint">${verdict[1]}</div>
        <ol>${names.map(n => `<li>${esc(n)}</li>`).join('')}</ol>
      </div>
      <div class="foot">${m.reviews.toLocaleString()} reviews ·
        cluster “${esc(leaf ? leaf.label : '')}” (${leaf ? leaf.confidence : '—'})</div>`;
  }

  const esc = (s) => String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---- frame ------------------------------------------------------------
  // Tuned by eye; see the note in scene.js on why this is a tint and not
  // a hull. Exposed on window.__atlas for adjustment without a rebuild.
  const DENSITY = { size: 62, alpha: 0.013 };
  const CARD_DEBOUNCE_MS = 80;
  let cardTimer = null;
  let hovered = -1;

  let labelStats = { drawn: 0, suppressed: 0, unlabelled: 0 };
  let frames = 0, fps = 0, lastT = performance.now();
  let firstPaint = 0;

  function drawFrame() {
    refreshColours();
    regl.poll();
    regl.clear({ color: [0.051, 0.067, 0.09, 1], depth: 1 });
    // Density tint underneath, then the points. The tint gives territories
    // something for the eye to hold onto; at purity 0.52 a hard hull would
    // assert a boundary the data does not support, so this asserts nothing
    // beyond where the members happen to be.
    const boost = scene.count > 20000 ? 0.9 : 1.25;
    scene.draw({ view, dim: 0, density: 1,
                 densitySize: DENSITY.size, densityAlpha: DENSITY.alpha });
    scene.draw({ view, dim: focus >= 0 ? 1 : 0, pointBoost: boost });

    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const entries = H.visibleLabels(project, overlay.width, overlay.height);
    labelStats = drawLabels(ctx, entries, project, dpr, colourOfCategory);

    if (!firstPaint && scene.count) firstPaint = performance.now() - t_start;

    frames++;
    const now = performance.now();
    if (now - lastT > 500) {
      fps = Math.round(frames * 1000 / (now - lastT));
      frames = 0; lastT = now;
    }
    $('hud').innerHTML =
        `<b>${scene.count.toLocaleString()}</b> <span>of `
      + `${manifest.n_points.toLocaleString()} shown</span>`
      + ` &nbsp;<b>z${tiles.levelFor(view.scale, home.scale)}</b>`
      + ` <span>${tiles.cache.size} tiles</span>`
      + ` &nbsp;<b>${(view.scale / home.scale).toFixed(1)}×</b>`
      + ` &nbsp;<b>${labelStats.drawn}</b> <span>labels</span>`
      + ` &nbsp;<b>${fps || '—'}</b> <span>fps</span>`;
  }

  // A label is tinted with the hue of the territory it belongs to, so it
  // reads as naming a region rather than floating over one.
  function colourOfCategory(cat) {
    const top = H.topAncestor(cat.id);
    const base = H.topIndex.get(top) ?? 0;
    const idx = (cat.id === top)
      ? base
      : base + 1 + ((cat.size ?? cat.id) % 5);
    return colourFor(idx);
  }

  function frame() { drawFrame(); requestAnimationFrame(frame); }

  // ---- input ------------------------------------------------------------
  // Loading mid-drag causes stutter, so panning is debounced; a zoom step is
  // discrete and the user is asking for new detail, so it is not.
  let dragTimer = null;
  const debouncedRefresh = () => {
    clearTimeout(dragTimer);
    dragTimer = setTimeout(() => refreshTiles(), 90);
  };

  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener('mousedown', e => {
    dragging = true; lx = e.clientX; ly = e.clientY;
  });
  window.addEventListener('mouseup', () => {
    if (dragging) { dragging = false; refreshTiles(); }
  });
  window.addEventListener('mousemove', e => {
    if (dragging) {
      view.cx -= (e.clientX - lx) / canvas.clientWidth * 2 / view.scale * aspect();
      view.cy += (e.clientY - ly) / canvas.clientHeight * 2 / view.scale;
      lx = e.clientX; ly = e.clientY;
      debouncedRefresh();
      return;
    }
    if (!grid) return;
    const r = canvas.getBoundingClientRect();
    cursor = [e.clientX, e.clientY];
    const [wx, wy] = unproject((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
    const radius = 10 * dpr / view.scale / overlay.width * 2 * aspect();
    const slot = grid.nearest(wx, wy, radius * 2);
    const gid = slot >= 0 ? scene.points.id[slot] : -1;

    // A point under the cursor is a link, so it should look like one.
    canvas.style.cursor = gid >= 0 ? 'pointer' : 'crosshair';
    hovered = gid;

    if (gid !== focus) {
      // Halo immediately -- it is already-loaded data and feels instant --
      // but debounce the card, which fetches a header image. Dragging the
      // cursor across the map would otherwise fire a request per pixel.
      setFocus(gid, false);
      clearTimeout(cardTimer);
      if (gid >= 0) {
        cardTimer = setTimeout(() => {
          if (hovered !== gid || focus !== gid) return;
          ensureKnn().then(k => showCardFor(gid, Array.from(k.neighboursOf(gid))));
        }, CARD_DEBOUNCE_MS);
      } else {
        card.hide();
      }
    } else if (!card.el.hidden) {
      card.place(cursor[0], cursor[1], []);
    }
  });
  canvas.addEventListener('mouseleave', () => {
    clearTimeout(cardTimer);
    hovered = -1;
    canvas.style.cursor = 'crosshair';
    setFocus(-1);
  });

  // Click opens the Steam page. Middle-click and ctrl/cmd-click are left to
  // the browser's own link handling by constructing a real anchor rather than
  // calling window.open, so modifier behaviour matches every other link.
  function openStore(e) {
    if (hovered < 0) return;
    const m = meta.get(hovered);
    if (!m) return;
    const a = document.createElement('a');
    a.href = STORE_URL(m.appid);
    a.target = '_blank';
    a.rel = 'noopener';
    a.dispatchEvent(new MouseEvent('click', {
      ctrlKey: e.ctrlKey, metaKey: e.metaKey, shiftKey: e.shiftKey,
      button: e.button, bubbles: false,
    }));
  }
  canvas.addEventListener('click', e => { if (e.button === 0) openStore(e); });
  canvas.addEventListener('auxclick', e => { if (e.button === 1) openStore(e); });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const [wx, wy] = unproject((e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr);
    const k = Math.exp(-e.deltaY * 0.0016);
    const next = Math.max(home.scale * 0.6, Math.min(home.scale * 3000, view.scale * k));
    view.cx = wx - (wx - view.cx) * (view.scale / next);
    view.cy = wy - (wy - view.cy) * (view.scale / next);
    view.scale = next;
    refreshTiles();
  }, { passive: false });
  window.addEventListener('resize', () => {
    resize(); home.scale = fitScale(); refreshTiles(true);
  });
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      // Escape belongs to the search box while it has focus: there it closes
      // the dropdown or blurs. stopPropagation in the box is not enough --
      // both handlers are on `window`, and stopPropagation does not stop
      // other listeners on the *same* element (that needs
      // stopImmediatePropagation), so whichever registered first still runs.
      // Checking focus is clearer than depending on registration order.
      if (document.activeElement === $('search')) return;
      clearTimeout(cardTimer);
      Object.assign(view, home); setFocus(-1); refreshTiles();
    }
  });

  await refreshTiles(true);
  if (status) status.remove();
  requestAnimationFrame(frame);

  // ---- fly-to -----------------------------------------------------------
  //
  // Zoom is multiplicative, so scale is interpolated geometrically: a linear
  // ramp crawls at the start and lurches at the end. Position uses the same
  // eased parameter so the target drifts to centre while being approached
  // rather than sliding first and then zooming.
  let flight = null;
  function flyTo(target, ms = 600) {
    const from = { cx: view.cx, cy: view.cy, scale: view.scale };
    const t0 = performance.now();
    flight = { from, target, t0, ms };
    return new Promise((done) => {
      const ease = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      let settled = false;
      const finish = (ok) => {
        if (settled) return;
        settled = true;
        clearTimeout(guard);
        if (ok) {
          view.scale = target.scale; view.cx = target.cx; view.cy = target.cy;
        }
        if (flight?.t0 === t0) flight = null;
        refreshTiles(true).then(() => done(ok));
      };

      // requestAnimationFrame is throttled to ~1Hz in a background tab, so a
      // flight started just before the tab is hidden would stall partway and
      // never settle -- leaving the camera between two places and any caller
      // awaiting it hung. The guard snaps to the destination instead. It is a
      // correctness backstop, not a timing mechanism: when rAF is running
      // normally the animation always finishes first.
      const guard = setTimeout(() => finish(true), ms + 250);

      function step(now) {
        if (settled) return;
        if (flight?.t0 !== t0) { settled = true; clearTimeout(guard); return done(false); }
        const t = Math.min(1, (now - t0) / ms);
        const k = ease(t);
        view.scale = from.scale * Math.pow(target.scale / from.scale, k);
        view.cx = from.cx + (target.cx - from.cx) * k;
        view.cy = from.cy + (target.cy - from.cy) * k;
        refreshTiles();
        if (t < 1) requestAnimationFrame(step);
        else finish(true);
      }
      requestAnimationFrame(step);
    });
  }

  function scaleToFit(bbox, margin = 1.35) {
    const w = Math.max(bbox[2] - bbox[0], 1e-6) * margin;
    const h = Math.max(bbox[3] - bbox[1], 1e-6) * margin;
    return Math.min(2 * aspect() / w, 2 / h);
  }

  async function flyToGame(r) {
    await flyTo({ cx: r.x, cy: r.y, scale: home.scale * 26 });
    // select after arrival so the halo and card appear on the destination
    // rather than streaking across the screen during the move
    await setFocus(r.index);
    const slot = scene.indexOfId(r.index);
    if (slot !== undefined) {
      const [sx, sy] = project(scene.points.x[slot], scene.points.y[slot]);
      cursor = [sx / dpr, sy / dpr];
      const knn = await ensureKnn();
      showCardFor(r.index, Array.from(knn.neighboursOf(r.index)));
    }
  }

  async function flyToLabel(r) {
    const b = r.bbox;
    await flyTo({ cx: (b[0] + b[2]) / 2, cy: (b[1] + b[3]) / 2,
                  scale: scaleToFit(b) });
  }

  const search = new SearchBox({
    input: $('search'), list: $('results'), hierarchy: H,
    onPickGame: flyToGame, onPickLabel: flyToLabel,
  });

  // ---- diagnostics ------------------------------------------------------
  const px4 = new Uint8Array(4);
  function benchFrames(n = 60) {
    const gl = regl._gl;
    scene.draw({ view, dim: 0 });
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px4);
    const t0 = performance.now();
    for (let i = 0; i < n; i++) scene.draw({ view, dim: 0 });
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px4);
    const ms = (performance.now() - t0) / n;
    return { msPerFrame: +ms.toFixed(3), impliedFps: Math.round(1000 / ms) };
  }

  async function findByName(name) {
    const shards = Math.ceil(manifest.n_points / manifest.meta_shard);
    for (let s = 0; s < shards; s++) {
      await meta.load(s * manifest.meta_shard);
      const shard = meta.shards.get(s);
      const k = shard.names.indexOf(name);
      if (k >= 0) return shard.from + k;
    }
    return -1;
  }

  installCapture(
    {
      view, home, scene, setFocus, refreshTiles, findByName,
      neighbours: async (gid) => {
        const knn = await ensureKnn();
        const nb = Array.from(knn.neighboursOf(gid));
        await meta.ensure([gid, ...nb]);
        return nb;
      },
      metaOf: (gid) => meta.get(gid),
      worldOf: (gid) => {
        const s = scene.indexOfId(gid);
        return s === undefined ? null : [scene.points.x[s], scene.points.y[s]];
      },
      pathOf: (gid) => {
        const s = scene.indexOfId(gid);
        if (s === undefined) return '';
        return H.chain(scene.points.cat[s])
          .map(id => H.byId.get(id).label).reverse().join('  ›  ');
      },
    },
    { canvas, overlay, drawFrame });

  window.__atlas = {
    DENSITY, card, images, showCardFor,
    H, view, home, scene, regl, tiles, meta, project, unproject,
    benchFrames, setFocus, refreshTiles, findByName,
    search, flyTo, flyToGame, flyToLabel, scaleToFit,
    labelStats: () => labelStats,
    timing: () => ({
      first_paint_ms: +firstPaint.toFixed(1),
      tile_requests: tiles.stats.requests,
      tile_bytes: tiles.stats.bytes,
      tile_hits: tiles.stats.hits,
      meta_requests: meta.stats.requests,
      meta_bytes: meta.stats.bytes,
    }),
    constants: { VISIBLE_MIN, VISIBLE_MAX, PRUNE_PURITY },
  };
  document.dispatchEvent(new CustomEvent('atlas-ready'));
}

boot().catch(err => {
  const s = document.getElementById('status');
  if (s) s.textContent = 'error: ' + err.message;
  console.error(err);
});
