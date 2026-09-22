// Label placement and styling.
//
// Collision: greedy in descending cluster size. Large territories claim their
// space first, so the labels that survive a crowded view are the ones naming
// the most games. Any label whose box overlaps one already placed is skipped
// entirely rather than nudged -- a nudged label points at the wrong region,
// which is worse than an absent one.
//
// Confidence tiers come from Phase 3 and decide *how* a label is drawn, not
// only whether:
//
//   strong  lift >= 4 over sibling categories   full weight
//   normal  lift >= 2                           full weight
//   weak    lift <  2                           dimmed italic -- the term is
//                                               true of the region but barely
//                                               distinguishes it
//   none    top term covers <25% of members     no label at all
//
// Dropping every uncertain label leaves grey continents that read as
// unfinished. Dimming instead tells the viewer "this region is loosely Casual
// games" while making the uncertainty visible.

const STYLE = {
  strong: { fill: '#f0f6fc', weight: 600, italic: false, alpha: 1.00 },
  normal: { fill: '#c9d1d9', weight: 500, italic: false, alpha: 0.95 },
  weak:   { fill: '#8b949e', weight: 400, italic: true,  alpha: 0.72 },
  none:   null,
};

export function styleFor(confidence) {
  return STYLE[confidence] ?? STYLE.normal;
}

function overlaps(a, b, pad = 3) {
  return !(a.x1 + pad < b.x0 || b.x1 + pad < a.x0 ||
           a.y1 + pad < b.y0 || b.y1 + pad < a.y0);
}

/**
 * Draw the visible labels, resolving collisions greedily.
 *
 * `entries` must already be sorted by descending size (Hierarchy does this).
 * Returns the number drawn and the number suppressed, which the HUD reports --
 * a high suppression count at a given zoom is the signal that the footprint
 * window needs adjusting, and is otherwise invisible.
 */
export function drawLabels(ctx, entries, project, dpr) {
  const placed = [];
  let drawn = 0, suppressed = 0, unlabelled = 0;

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (const { cat, opacity } of entries) {
    const style = styleFor(cat.confidence);
    if (!style) { unlabelled++; continue; }

    const [sx, sy] = project(cat.centroid[0], cat.centroid[1]);
    if (!Number.isFinite(sx) || !Number.isFinite(sy)) continue;

    // Cull off-screen labels before measuring them. This matters more than it
    // looks: leaves are exempt from the upper footprint cutoff (they have no
    // children to hand off to), so at deep zoom every leaf in the tree becomes
    // a candidate -- 876 of them at 45x, almost all far outside the viewport.
    // Culling first keeps the collision pass proportional to what is visible.
    const margin = 80 * dpr;
    if (sx < -margin || sy < -margin ||
        sx > ctx.canvas.width + margin || sy > ctx.canvas.height + margin) {
      continue;
    }

    // size the type by how much of the screen the category occupies, within
    // a range that stays readable at either end
    const px = Math.max(11, Math.min(20, 11 + Math.log2(cat.size) * 0.85));
    ctx.font = `${style.italic ? 'italic ' : ''}${style.weight} ${px}px `
             + `ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`;

    const text = cat.label;
    const w = ctx.measureText(text).width;
    const box = { x0: sx - w / 2, x1: sx + w / 2, y0: sy - px * 0.7, y1: sy + px * 0.7 };

    if (placed.some(p => overlaps(box, p))) { suppressed++; continue; }
    placed.push(box);

    ctx.globalAlpha = opacity * style.alpha;
    ctx.lineWidth = 3 * dpr;
    ctx.strokeStyle = 'rgba(13,17,23,0.85)';
    ctx.strokeText(text, sx, sy);
    ctx.fillStyle = style.fill;
    ctx.fillText(text, sx, sy);
    drawn++;
  }

  ctx.restore();
  return { drawn, suppressed, unlabelled };
}
