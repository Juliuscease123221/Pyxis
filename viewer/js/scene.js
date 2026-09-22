// WebGL point rendering via regl. One draw call for every visible point.
//
// The point set changes as tiles stream in, so buffers are allocated once at a
// generous capacity and refilled with `subdata` rather than reallocated per
// update -- reallocating a GPU buffer on every viewport change is the easiest
// way to turn a smooth pan into a stutter.
//
// Colour is an attribute rather than a uniform lookup because it changes only
// when the active ancestor level changes: a buffer update on zoom, not
// per-frame work. Highlight is a separate attribute so haloing a game's
// neighbours costs one update, not a second draw pass.

const PALETTE = [
  [0.40,0.76,0.95],[0.98,0.63,0.31],[0.47,0.83,0.53],[0.94,0.47,0.52],
  [0.71,0.62,0.93],[0.85,0.68,0.50],[0.96,0.60,0.82],[0.62,0.78,0.88],
  [0.90,0.86,0.40],[0.42,0.85,0.85],[0.60,0.55,0.95],[0.95,0.55,0.40],
  [0.55,0.80,0.70],[0.88,0.50,0.70],[0.50,0.70,0.90],[0.80,0.80,0.55],
  [0.70,0.90,0.60],[0.95,0.75,0.55],[0.60,0.65,0.85],[0.85,0.55,0.85],
  [0.45,0.78,0.72],[0.92,0.68,0.42],[0.66,0.84,0.44],[0.52,0.62,0.92],
  [0.88,0.62,0.62],[0.72,0.72,0.95],[0.58,0.88,0.80],
];

export function colourFor(i) { return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length]; }

export function createScene(regl, dpr, capacity = 80000) {
  const position = new Float32Array(capacity * 2);
  const importance = new Float32Array(capacity);
  const colour = new Float32Array(capacity * 3);
  const highlight = new Float32Array(capacity);

  const buffers = {
    position: regl.buffer({ data: position, usage: 'dynamic' }),
    importance: regl.buffer({ data: importance, usage: 'dynamic' }),
    colour: regl.buffer({ data: colour, usage: 'dynamic' }),
    highlight: regl.buffer({ data: highlight, usage: 'dynamic' }),
  };

  let current = { count: 0, id: new Uint32Array(0), cat: new Uint32Array(0) };
  let indexOfId = new Map();

  const draw = regl({
    vert: `
      precision highp float;
      attribute vec2 position;
      attribute vec3 colour;
      attribute float importance;
      attribute float highlight;
      uniform vec2 center;
      uniform float scale, aspect, dpr, dim, pointBoost, density, densitySize;
      varying vec3 vColour;
      varying float vHighlight;
      varying float vAlpha;
      void main() {
        vec2 p = (position - center) * scale;
        p.x /= aspect;
        gl_Position = vec4(p, 0.0, 1.0);
        float base = (1.7 + importance * 3.4) * pointBoost;
        float boost = highlight > 1.5 ? 3.6 : (highlight > 0.5 ? 2.4 : 0.0);
        // The density pass draws the same points far larger and very faint,
        // additively, so overlapping members of a territory accumulate into
        // a soft tint. It gives the eye a region to hold onto without
        // drawing a boundary the data does not support -- purity is 0.52 at
        // depth 1, so any hard hull would be asserting more than is true.
        gl_PointSize = density > 0.5 ? (densitySize * dpr) : (base + boost) * dpr;
        vColour = colour;
        vHighlight = highlight;
        vAlpha = highlight > 0.5 ? 1.0 : mix(0.92, 0.38, dim);
      }`,
    frag: `
      precision highp float;
      uniform float density, densityAlpha;
      varying vec3 vColour;
      varying float vHighlight;
      varying float vAlpha;
      void main() {
        vec2 d = gl_PointCoord - vec2(0.5);
        float r2 = dot(d, d);
        if (r2 > 0.25) discard;
        if (density > 0.5) {
          float g = smoothstep(0.25, 0.0, r2);
          gl_FragColor = vec4(vColour * g * densityAlpha, 1.0);
          return;
        }
        float core = smoothstep(0.25, 0.05, r2);
        if (vHighlight > 0.5) {
          float ring = smoothstep(0.25, 0.19, r2) - smoothstep(0.15, 0.09, r2);
          vec3 c = vHighlight > 1.5 ? vec3(1.0, 0.95, 0.55) : vec3(1.0, 1.0, 1.0);
          gl_FragColor = vec4(mix(vColour, c, 0.68), min(1.0, core + ring * 1.5));
          return;
        }
        gl_FragColor = vec4(vColour, core * vAlpha);
      }`,
    attributes: {
      position: buffers.position,
      colour: buffers.colour,
      importance: buffers.importance,
      highlight: buffers.highlight,
    },
    uniforms: {
      center: (_, p) => [p.view.cx, p.view.cy],
      scale: (_, p) => p.view.scale,
      aspect: ({ viewportWidth, viewportHeight }) => viewportWidth / viewportHeight,
      dim: (_, p) => p.dim ?? 0,
      pointBoost: (_, p) => p.pointBoost ?? 1,
      density: (_, p) => p.density ?? 0,
      densitySize: (_, p) => p.densitySize ?? 62,
      densityAlpha: (_, p) => p.densityAlpha ?? 0.013,
      dpr,
    },
    count: () => current.count,
    primitive: 'points',
    blend: {
      enable: true,
      func: (_, p) => p.density
        ? { srcRGB: 'one', srcAlpha: 'one', dstRGB: 'one', dstAlpha: 'one' }
        : { srcRGB: 'src alpha', srcAlpha: 1,
            dstRGB: 'one minus src alpha', dstAlpha: 1 },
    },
    depth: { enable: false },
  });

  return {
    draw,
    get count() { return current.count; },
    get points() { return current; },
    indexOfId: (id) => indexOfId.get(id),

    /** Replace the visible point set (called when the tile set changes). */
    setPoints(P) {
      const n = Math.min(P.count, capacity);
      for (let i = 0; i < n; i++) {
        position[i * 2] = P.x[i];
        position[i * 2 + 1] = P.y[i];
        importance[i] = P.imp[i] / 65535;
      }
      buffers.position.subdata(position.subarray(0, n * 2));
      buffers.importance.subdata(importance.subarray(0, n));
      highlight.fill(0, 0, n);
      buffers.highlight.subdata(highlight.subarray(0, n));
      current = { count: n, id: P.id, cat: P.cat, x: P.x, y: P.y };
      indexOfId = new Map();
      for (let i = 0; i < n; i++) indexOfId.set(P.id[i], i);
      return n;
    },

    setColours(indexFn) {
      const n = current.count;
      for (let i = 0; i < n; i++) {
        const c = colourFor(indexFn(i));
        colour[i * 3] = c[0]; colour[i * 3 + 1] = c[1]; colour[i * 3 + 2] = c[2];
      }
      buffers.colour.subdata(colour.subarray(0, n * 3));
    },

    clearHighlight() {
      highlight.fill(0, 0, current.count);
      buffers.highlight.subdata(highlight.subarray(0, current.count));
    },

    /** `focusId` and `neighbourIds` are global point ids, not buffer slots. */
    setHighlight(focusId, neighbourIds) {
      const n = current.count;
      highlight.fill(0, 0, n);
      for (const gid of neighbourIds) {
        const k = indexOfId.get(gid);
        if (k !== undefined) highlight[k] = 1;
      }
      const f = indexOfId.get(focusId);
      if (f !== undefined) highlight[f] = 2;
      buffers.highlight.subdata(highlight.subarray(0, n));
    },
  };
}
