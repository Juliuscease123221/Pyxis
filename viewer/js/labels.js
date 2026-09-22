// Label placement and styling.
//
// Collision: greedy in descending cluster size. Large territories claim space
// first, so the labels that survive a crowded view name the most games. A
// label whose box overlaps one already placed is skipped rather than nudged --
// a nudged label points at the wrong region, which is worse than an absent one.
//
// Confidence tiers come from Phase 3 and decide *how* a label is drawn:
//
//   strong  lift >= 4 over sibling categories   full weight
//   normal  lift >= 2                           full weight
//   weak    lift <  2                           dimmed italic
//   none    top term covers <25% of members     no label
//
// Dropping every uncertain label leaves grey continents that read as
// unfinished. Dimming instead says "this region is loosely Casual games" while
// keeping the uncertainty visible.

const STYLE = {
  strong: { weight: 600, italic: false, alpha: 1.00, tint: 0.55, light: 0.86 },
  normal: { weight: 500, italic: false, alpha: 0.94, tint: 0.50, light: 0.78 },
  weak:   { weight: 400, italic: true,  alpha: 0.70, tint: 0.62, light: 0.58 },
  none:   null,
};

export function styleFor(confidence) {
  return STYLE[confidence] ?? STYLE.normal;
}

/**
 * How many terms of a label to show, from its screen footprint.
 *
 * Every label carrying three slash-separated terms is unreadable at a glance
 * and visually uniform, so nothing stands out. A label that occupies very
 * little of the screen gets its single most distinctive term; as its region
 * grows there is room to qualify it. The thresholds are in the same units as
 * the LOD window (fraction of viewport), so a label gains terms over the same
 * interval it fades in.
 */
export function termsForFootprint(f) {
  if (f > 0.055) return 3;
  if (f > 0.014) return 2;
  return 1;
}

function labelText(cat, n) {
  const terms = (cat.terms && cat.terms.length)
    ? cat.terms
    : String(cat.label || '').split(' / ');
  return terms.slice(0, n).join(' / ');
}

function overlaps(a, b, pad = 3) {
  return !(a.x1 + pad < b.x0 || b.x1 + pad < a.x0 ||
           a.y1 + pad < b.y0 || b.y1 + pad < a.y0);
}

function toCss(rgb, light) {
  // Pull the cluster's colour toward white so text stays legible on the dark
  // field while still reading as belonging to its region.
  const m = (v) => Math.round(255 * (v + (1 - v) * light));
  return `rgb(${m(rgb[0])},${m(rgb[1])},${m(rgb[2])})`;
}

/**
 * Draw the visible labels, resolving collisions greedily.
 *
 * `entries` must already be sorted by descending size (Hierarchy does this).
 * `colourOf(cat)` returns the cluster's palette colour so the label can be
 * tinted to match its region -- without it labels float free of the blobs they
 * name, and confidence has only opacity to signal with.
 *
 * `layerAlpha` multiplies every label's opacity, which is how the `L` toggle
 * fades the whole layer. At zero the painting is skipped but the *placement*
 * still runs: the returned counts stay live while the layer is hidden, and
 * more importantly nothing about the LOD or collision state has to be rebuilt
 * when it comes back, so re-enabling is correct for the current viewport on
 * the very next frame rather than one frame late.
 */
export function drawLabels(ctx, entries, project, dpr, colourOf, layerAlpha = 1) {
  const placed = [];
  let drawn = 0, suppressed = 0, unlabelled = 0;
  const paint = layerAlpha > 0.001;

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (const { cat, opacity, footprint } of entries) {
    const style = styleFor(cat.confidence);
    if (!style) { unlabelled++; continue; }

    const [sx, sy] = project(cat.centroid[0], cat.centroid[1]);
    if (!Number.isFinite(sx) || !Number.isFinite(sy)) continue;

    // Cull off-screen labels before measuring them. Leaves are exempt from the
    // upper footprint cutoff (they have no children to hand off to), so at
    // deep zoom every leaf in the tree becomes a candidate -- 876 of them at
    // 45x, almost all far outside the viewport.
    const margin = 90 * dpr;
    if (sx < -margin || sy < -margin ||
        sx > ctx.canvas.width + margin || sy > ctx.canvas.height + margin) {
      continue;
    }

    const text = labelText(cat, termsForFootprint(footprint));
    const px = Math.max(11, Math.min(21, 11 + Math.log2(Math.max(cat.size, 2)) * 0.9)) * dpr;
    ctx.font = `${style.italic ? 'italic ' : ''}${style.weight} ${px}px `
             + `ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`;

    const w = ctx.measureText(text).width;
    const box = { x0: sx - w / 2, x1: sx + w / 2,
                  y0: sy - px * 0.7, y1: sy + px * 0.7 };

    if (placed.some(p => overlaps(box, p))) { suppressed++; continue; }
    placed.push(box);

    if (paint) {
      ctx.globalAlpha = opacity * style.alpha * layerAlpha;
      ctx.lineWidth = 3.4 * dpr;
      ctx.strokeStyle = 'rgba(13,17,23,0.92)';
      ctx.strokeText(text, sx, sy);
      const rgb = colourOf ? colourOf(cat) : null;
      ctx.fillStyle = rgb ? toCss(rgb, style.light) : '#e6edf3';
      ctx.fillText(text, sx, sy);
    }
    drawn++;
  }

  ctx.restore();
  return { drawn, suppressed, unlabelled };
}
