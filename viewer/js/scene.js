// WebGL point rendering via regl. One draw call for every point.
//
// Colour is an attribute rather than a uniform lookup because it changes only
// when the active ancestor level changes -- a buffer subdata on zoom, not
// per-frame work. Highlight state is a separate float attribute so haloing a
// game's neighbours costs one buffer update, not a second draw pass.

const PALETTE = [
  [0.40,0.76,0.95],[0.98,0.63,0.31],[0.47,0.83,0.53],[0.94,0.47,0.52],
  [0.71,0.62,0.93],[0.85,0.68,0.50],[0.96,0.60,0.82],[0.62,0.78,0.88],
  [0.90,0.86,0.40],[0.42,0.85,0.85],[0.60,0.55,0.95],[0.95,0.55,0.40],
  [0.55,0.80,0.70],[0.88,0.50,0.70],[0.50,0.70,0.90],[0.80,0.80,0.55],
  [0.70,0.90,0.60],[0.95,0.75,0.55],[0.60,0.65,0.85],[0.85,0.55,0.85],
  [0.45,0.78,0.72],[0.92,0.68,0.42],[0.66,0.84,0.44],[0.52,0.62,0.92],
  [0.88,0.62,0.62],[0.72,0.72,0.95],[0.58,0.88,0.80],
];

export function colourFor(i) { return PALETTE[i % PALETTE.length]; }

export function createScene(regl, P, dpr) {
  const position = new Float32Array(P.count * 2);
  for (let i = 0; i < P.count; i++) {
    position[i * 2] = P.x[i];
    position[i * 2 + 1] = P.y[i];
  }
  const importance = new Float32Array(P.count);
  for (let i = 0; i < P.count; i++) importance[i] = P.imp[i] / 65535;

  const colour = new Float32Array(P.count * 3);
  const highlight = new Float32Array(P.count);   // 0 normal, 1 neighbour, 2 focus

  const buffers = {
    position: regl.buffer(position),
    importance: regl.buffer(importance),
    colour: regl.buffer({ data: colour, usage: 'dynamic' }),
    highlight: regl.buffer({ data: highlight, usage: 'dynamic' }),
  };

  const draw = regl({
    vert: `
      precision highp float;
      attribute vec2 position;
      attribute vec3 colour;
      attribute float importance;
      attribute float highlight;
      uniform vec2 center;
      uniform float scale, aspect, dpr, dim;
      varying vec3 vColour;
      varying float vHighlight;
      varying float vAlpha;
      void main() {
        vec2 p = (position - center) * scale;
        p.x /= aspect;
        gl_Position = vec4(p, 0.0, 1.0);
        float base = 1.5 + importance * 3.2;
        // focused point and its neighbours grow so they are findable anywhere
        float boost = highlight > 1.5 ? 3.4 : (highlight > 0.5 ? 2.2 : 0.0);
        gl_PointSize = (base + boost) * dpr;
        vColour = colour;
        vHighlight = highlight;
        // when something is focused, everything else recedes
        vAlpha = highlight > 0.5 ? 1.0 : mix(0.92, 0.38, dim);
      }`,
    frag: `
      precision highp float;
      varying vec3 vColour;
      varying float vHighlight;
      varying float vAlpha;
      void main() {
        vec2 d = gl_PointCoord - vec2(0.5);
        float r2 = dot(d, d);
        if (r2 > 0.25) discard;
        float core = smoothstep(0.25, 0.05, r2);
        if (vHighlight > 0.5) {
          // ring so a highlighted point reads even against a dense field
          float ring = smoothstep(0.25, 0.19, r2) - smoothstep(0.15, 0.09, r2);
          vec3 c = vHighlight > 1.5 ? vec3(1.0, 0.95, 0.55) : vec3(1.0, 1.0, 1.0);
          gl_FragColor = vec4(mix(vColour, c, 0.65), min(1.0, core + ring * 1.4));
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
      dpr,
    },
    count: P.count,
    primitive: 'points',
    blend: {
      enable: true,
      func: { srcRGB: 'src alpha', srcAlpha: 1,
              dstRGB: 'one minus src alpha', dstAlpha: 1 },
    },
    depth: { enable: false },
  });

  return {
    draw,
    colour,
    highlight,
    /** Recolour by active ancestor. Called only when the level changes. */
    setColours(indexOf) {
      for (let i = 0; i < P.count; i++) {
        const c = colourFor(indexOf(i));
        colour[i * 3] = c[0]; colour[i * 3 + 1] = c[1]; colour[i * 3 + 2] = c[2];
      }
      buffers.colour.subdata(colour);
    },
    clearHighlight() {
      highlight.fill(0);
      buffers.highlight.subdata(highlight);
    },
    setHighlight(focus, neighbours) {
      highlight.fill(0);
      for (const n of neighbours) highlight[n] = 1;
      if (focus >= 0) highlight[focus] = 2;
      buffers.highlight.subdata(highlight);
    },
  };
}
