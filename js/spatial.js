// A uniform grid for "which point is under the cursor".
//
// SPEC.md asks for a spatial lookup rather than a linear scan, and at 56k
// points a scan per mousemove is 56k distance tests at 60Hz. A uniform grid is
// the right structure here rather than a k-d tree: the query is always a small
// radius around a cursor, the data is static, and build cost is one pass.
//
// Note this answers *screen* proximity, which is only ever used for hit
// testing. It is never used to decide what is similar to what -- only 10-15%
// of a game's true neighbours are among its nearest on screen, so similarity
// comes from the precomputed high-dimensional lists in knn.bin.

export class Grid {
  constructor(x, y, bbox, cells = 128) {
    this.x = x; this.y = y;
    const [minx, miny, maxx, maxy] = bbox;
    this.minx = minx; this.miny = miny;
    this.w = Math.max(maxx - minx, 1e-9);
    this.h = Math.max(maxy - miny, 1e-9);
    this.n = cells;

    const counts = new Int32Array(cells * cells + 1);
    const cellOf = new Int32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const cx = Math.min(cells - 1, Math.max(0, ((x[i] - minx) / this.w * cells) | 0));
      const cy = Math.min(cells - 1, Math.max(0, ((y[i] - miny) / this.h * cells) | 0));
      const c = cy * cells + cx;
      cellOf[i] = c;
      counts[c + 1]++;
    }
    for (let c = 0; c < cells * cells; c++) counts[c + 1] += counts[c];
    const items = new Int32Array(x.length);
    const cursor = counts.slice(0, cells * cells);
    for (let i = 0; i < x.length; i++) items[cursor[cellOf[i]]++] = i;

    this.starts = counts;
    this.items = items;
  }

  /** Nearest point to (wx, wy) within `radius` world units, or -1. */
  nearest(wx, wy, radius) {
    const cells = this.n;
    const cx = ((wx - this.minx) / this.w * cells) | 0;
    const cy = ((wy - this.miny) / this.h * cells) | 0;
    const rx = Math.max(1, Math.ceil(radius / this.w * cells));
    const ry = Math.max(1, Math.ceil(radius / this.h * cells));

    let best = -1, bestD = radius * radius;
    for (let gy = Math.max(0, cy - ry); gy <= Math.min(cells - 1, cy + ry); gy++) {
      for (let gx = Math.max(0, cx - rx); gx <= Math.min(cells - 1, cx + rx); gx++) {
        const c = gy * cells + gx;
        for (let k = this.starts[c]; k < this.starts[c + 1]; k++) {
          const i = this.items[k];
          const dx = this.x[i] - wx, dy = this.y[i] - wy;
          const d = dx * dx + dy * dy;
          if (d < bestD) { bestD = d; best = i; }
        }
      }
    }
    return best;
  }
}
