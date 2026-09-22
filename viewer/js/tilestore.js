// Streaming tile loader: which tiles the viewport needs, fetched and cached.
//
// This is the consumer for the `tiles` library. The pyramid is built by
// `python -m viewer.prepare` (which calls `tiles.quadtree.build`), served as
// static files, and pulled in here a viewport at a time.
//
// Design notes that matter for it feeling smooth rather than correct:
//
// * **Render the zoom level, not the union.** A point appears in a coarse tile
//   and again in finer ones -- that redundancy is what makes each tile
//   independently renderable. Drawing every cached tile would therefore draw
//   the same point several times. Only tiles at the current level are drawn.
//
// * **Keep showing what you have.** When the level changes, the previous
//   level's points stay on screen until the new ones arrive, so a zoom never
//   flashes empty. Tiles already cached render immediately.
//
// * **Debounce during drags, not during zooms.** Loading mid-drag causes
//   stutter, and a drag produces a continuous stream of viewport changes.
//   A zoom step is discrete and the user expects new detail, so it is not
//   debounced.

import { decodeTile } from './binary.js';

export class TileStore {
  constructor({ bounds, maxZoom, base = 'public/tiles', capacity = 200 }) {
    this.bounds = bounds;
    this.maxZoom = maxZoom;
    this.base = base;
    this.capacity = capacity;
    this.cache = new Map();       // key -> decoded tile   (Map = insertion order = LRU)
    this.inflight = new Map();    // key -> Promise
    this.stats = { requests: 0, hits: 0, bytes: 0, evictions: 0 };
  }

  /**
   * Zoom level to request for a given view scale.
   *
   * The pyramid doubles resolution per level, so the level whose tiles are
   * about one screen wide is `log2` of how many times the data extent divides
   * into the viewport.
   */
  levelFor(scale, homeScale) {
    const z = Math.floor(Math.log2(Math.max(scale / homeScale, 1)) + 0.4);
    return Math.max(0, Math.min(this.maxZoom, z));
  }

  tilesFor(z, view, aspect) {
    const [minx, miny, maxx, maxy] = this.bounds;
    const n = 1 << z;
    const w = Math.max(maxx - minx, 1e-12);
    const h = Math.max(maxy - miny, 1e-12);

    // visible world rectangle
    const halfW = aspect / view.scale;
    const halfH = 1 / view.scale;
    const vx0 = view.cx - halfW, vx1 = view.cx + halfW;
    const vy0 = view.cy - halfH, vy1 = view.cy + halfH;

    const x0 = Math.max(0, Math.min(n - 1, Math.floor((vx0 - minx) / w * n)));
    const x1 = Math.max(0, Math.min(n - 1, Math.floor((vx1 - minx) / w * n)));
    const y0 = Math.max(0, Math.min(n - 1, Math.floor((vy0 - miny) / h * n)));
    const y1 = Math.max(0, Math.min(n - 1, Math.floor((vy1 - miny) / h * n)));

    const out = [];
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) out.push(`${z}/${x}/${y}`);
    }
    return out;
  }

  get(key) {
    const t = this.cache.get(key);
    if (t) {                       // refresh LRU position
      this.cache.delete(key);
      this.cache.set(key, t);
      this.stats.hits++;
    }
    return t;
  }

  async load(key) {
    const hit = this.cache.get(key);
    if (hit) { this.stats.hits++; return hit; }
    if (this.inflight.has(key)) return this.inflight.get(key);

    const p = (async () => {
      this.stats.requests++;
      const res = await fetch(`${this.base}/${key}.bin`);
      if (!res.ok) { this.cache.set(key, null); return null; }  // empty quadrant
      const buf = await res.arrayBuffer();
      this.stats.bytes += buf.byteLength;
      const tile = decodeTile(buf);
      this.cache.set(key, tile);
      while (this.cache.size > this.capacity) {
        const oldest = this.cache.keys().next().value;
        this.cache.delete(oldest);
        this.stats.evictions++;
      }
      return tile;
    })().finally(() => this.inflight.delete(key));

    this.inflight.set(key, p);
    return p;
  }

  /** Everything already cached for these keys, for immediate drawing. */
  cached(keys) {
    const out = [];
    for (const k of keys) {
      const t = this.get(k);
      if (t) out.push(t);
    }
    return out;
  }

  async ensure(keys) {
    await Promise.all(keys.map(k => this.load(k)));
    return this.cached(keys);
  }
}

/**
 * Concatenate a set of tiles into flat buffers the renderer can upload.
 *
 * Rebuilt only when the visible tile set changes, not per frame.
 */
export function flatten(tiles) {
  let n = 0;
  for (const t of tiles) n += t.count;
  const x = new Float32Array(n), y = new Float32Array(n);
  const id = new Uint32Array(n), cat = new Uint32Array(n);
  const imp = new Uint16Array(n);
  let o = 0;
  for (const t of tiles) {
    x.set(t.x, o); y.set(t.y, o); id.set(t.id, o);
    cat.set(t.cat, o); imp.set(t.imp, o);
    o += t.count;
  }
  return { count: n, x, y, id, cat, imp };
}
