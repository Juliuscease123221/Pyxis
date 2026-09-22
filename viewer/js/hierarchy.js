// The cluster hierarchy, and the level-of-detail rule built on it.
//
// ---------------------------------------------------------------------------
// Why level of detail is keyed to screen footprint rather than tree depth
// ---------------------------------------------------------------------------
//
// SPEC.md argues for footprint LOD from *depth variance*: dense regions nest
// six or seven levels, sparse ones stop at two, so any depth-keyed threshold
// leaves half the map unlabelled.
//
// That argument does not apply to the tree that shipped. Recursive Leiden
// produces near-uniform depth: 83% of games sit at depth 5 and 17% at depth 4.
// Nothing nests to seven and nothing stops at two.
//
// The real reason is *size* variance at a fixed depth. The 27 top-level
// territories run from 276 games (Hidden Object) to 7,239 (Visual Novel) -- a
// 26x spread -- and across the whole tree sizes run 16 to 7,239 with a median
// of 62. In screen terms the trimmed extents of depth-1 territories span 1.7%
// to 14.6% of the map, an order of magnitude apart at the same depth.
//
// A depth-keyed rule shows all 27 depth-1 labels at one zoom. At the zoom
// where Visual Novel is comfortably readable, Hidden Object is an unreadable
// speck; at the zoom where Hidden Object is legible, Visual Novel has long
// since needed subdividing. Footprint keys the decision to what the viewer can
// actually see, so small territories surface later and large ones subdivide
// sooner, with no per-level tuning anywhere.
//
// ---------------------------------------------------------------------------
// Robust extents
// ---------------------------------------------------------------------------
//
// LOD uses `bbox_core` (5th-95th percentile), not `bbox`. A raw box is set by
// a category's two most extreme members, so one stray game stretches it across
// the map: raw boxes cover 48-75% of the map where trimmed ones cover 1.7-14.6%,
// a 24-29x difference. Using raw boxes, every territory exceeds the "too large
// to label" cutoff permanently and no label ever appears.

// Visibility window, as a fraction of the viewport the category's trimmed
// extent covers. Calibrated against measured footprints rather than guessed:
// at the home view the 16 surviving territories cover 0.08% to 1.56% of a
// 1440x900 screen (median 0.45%). An earlier guess of 0.05 was an order of
// magnitude too high and labelled nothing but the scattered territories,
// because those are the only ones with boxes that large.
export const VISIBLE_MIN = 0.002;  // below this, too small to be worth naming
export const VISIBLE_MAX = 0.25;   // above this, zoomed in far enough for children

// A territory whose points are scattered is not a place, so it is not drawn as
// one labelled region -- its children are surfaced instead.
export const PRUNE_PURITY = 0.35;

export const PRUNE_NOTE = `
Territories below purity 0.35 are not drawn as a single labelled region; their
children are drawn instead. 11 of 27 top-level territories are pruned.

The threshold is insensitive. Sorted by purity the territories fall into two
groups with a clear gap between them: pruned runs Education 0.03, VR 0.04,
Survival 0.05, Free to Play 0.06, RPG 0.08, Rhythm 0.10, Sexual Content 0.10,
RTS 0.12, Indie 0.12, Puzzle 0.13, Online Co-Op 0.26; kept starts at Shooter
0.38, then Local Multiplayer 0.53, Point & Click 0.61, Management 0.69, Horror
0.79, Platformer 0.88. Any threshold in 0.27-0.37 selects exactly the same
split, so 0.35 is not a tuned number.

The rule is load-bearing rather than cosmetic, which only became clear from
measurement: footprint correlates *inversely* with purity, because a scattered
category has a large bounding box precisely because its members are spread
out. At the home view the six largest footprints are RPG (purity 0.08),
Education (0.03), VR (0.04), Puzzle (0.13), Free to Play (0.06) and Indie
(0.12) -- every one of them a territory that is not a place. Without pruning,
footprint LOD would preferentially label the worst categories on the map.

Judgement calls worth flagging: RTS (0.12) and Survival (0.05) are real genres
that a viewer would expect to see named, and they are pruned. Their members
genuinely are scattered in this layout, so the honest result is to name their
sub-regions instead. If a later layout makes them cohesive they will return
without any change here.`.trim();

export function smoothstep(a, b, x) {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

export class Hierarchy {
  constructor(manifest) {
    this.byId = new Map(manifest.categories.map(c => [c.id, c]));
    this.root = manifest.categories.find(c => c.parent === null);
    this.children = new Map();
    for (const c of manifest.categories) {
      if (c.parent === null) continue;
      if (!this.children.has(c.parent)) this.children.set(c.parent, []);
      this.children.get(c.parent).push(c.id);
    }
    this._ancestorCache = new Map();
    this.topLevel = (this.children.get(this.root.id) || []).slice()
      .sort((a, b) => this.byId.get(b).size - this.byId.get(a).size);
    this.topIndex = new Map(this.topLevel.map((id, i) => [id, i]));
  }

  // leaf id -> [leaf, ..., depth-1 ancestor]
  chain(id) {
    if (this._ancestorCache.has(id)) return this._ancestorCache.get(id);
    const out = [];
    let cur = this.byId.get(id), guard = 0;
    while (cur && cur.parent !== null && guard++ < 64) {
      out.push(cur.id);
      cur = this.byId.get(cur.parent);
    }
    this._ancestorCache.set(id, out);
    return out;
  }

  ancestorAtDepth(id, depth) {
    for (const nid of this.chain(id)) {
      const c = this.byId.get(nid);
      if (c && c.depth === depth) return nid;
    }
    return this.topLevel.length ? this.chain(id).at(-1) ?? this.root.id : this.root.id;
  }

  topAncestor(id) {
    const ch = this.chain(id);
    return ch.length ? ch[ch.length - 1] : this.root.id;
  }

  area(c) {
    const b = c.bbox_core || c.bbox;
    return Math.max(b[2] - b[0], 1e-9) * Math.max(b[3] - b[1], 1e-9);
  }

  // Is this node pruned as "not a place"?
  isPruned(c) {
    return c.purity !== null && c.purity !== undefined
        && c.purity < PRUNE_PURITY
        && (this.children.get(c.id) || []).length > 0;
  }

  /**
   * Fraction of the viewport a category covers, in screen pixels.
   *
   * Computed by projecting the trimmed box's corners rather than scaling a
   * world-space area by `scale^2`. The clip-space version has to carry an
   * aspect correction, and getting that correction wrong is silent: labels
   * simply select the wrong nodes, which looks like a tuning problem rather
   * than an arithmetic one. Pixels are unambiguous and directly comparable to
   * the thresholds, which are stated as fractions of the screen.
   */
  footprint(c, project, vw, vh) {
    const b = c.bbox_core || c.bbox;
    const [x0, y0] = project(b[0], b[1]);
    const [x1, y1] = project(b[2], b[3]);
    const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0);
    return (w * h) / Math.max(vw * vh, 1);
  }

  /**
   * Which labels to draw, and how strongly.
   *
   * Opacity comes from the same smoothstep that governs visibility, so a
   * parent fades out over exactly the interval its children fade in -- the
   * cross-fade is one rule, not two that have to be kept in sync.
   */
  visibleLabels(project, vw, vh) {
    const out = [];
    const walk = (id) => {
      const c = this.byId.get(id);
      if (!c) return;
      const f = this.footprint(c, project, vw, vh);
      const pruned = this.isPruned(c);

      if (!pruned && f > VISIBLE_MIN) {
        const fadeIn = smoothstep(VISIBLE_MIN, VISIBLE_MIN * 2.5, f);
        // The upper cutoff exists so a category hands off to its children once
        // it fills the screen. A leaf has no children to hand off to, so
        // applying it there just deletes the label: at the end of a deep zoom
        // the leaf's footprint passes 1.0 and the map goes silent exactly
        // where it should be most specific. Leaves therefore never fade out.
        const isLeaf = (this.children.get(id) || []).length === 0;
        const fadeOut = isLeaf
          ? 1
          : 1 - smoothstep(VISIBLE_MAX * 0.7, VISIBLE_MAX, f);
        const opacity = Math.min(fadeIn, fadeOut);
        if (opacity > 0.02) out.push({ cat: c, footprint: f, opacity });
      }
      // recurse while the node is still big enough for children to matter,
      // and always through a pruned node so its children can stand in for it
      if (pruned || f > VISIBLE_MIN * 0.6) {
        for (const k of this.children.get(id) || []) walk(k);
      }
    };
    for (const t of this.topLevel) walk(t);
    return out.sort((a, b) => b.cat.size - a.cat.size);
  }

  /**
   * The deepest ancestor level currently "active", used to colour points.
   *
   * Points recolour by active ancestor, so a territory visibly splits into
   * coloured sub-regions as you zoom without any point moving.
   */
  activeAncestorOf(leafId, project, vw, vh) {
    const ch = this.chain(leafId);
    for (let i = 0; i < ch.length; i++) {
      const c = this.byId.get(ch[i]);
      if (!c || this.isPruned(c)) continue;
      const f = this.footprint(c, project, vw, vh);
      const isLeaf = (this.children.get(ch[i]) || []).length === 0;
      // same leaf exemption as visibleLabels: past the upper cutoff a leaf has
      // nothing to defer to, so it stays the active level
      if (f > VISIBLE_MIN && (isLeaf || f < VISIBLE_MAX)) return ch[i];
    }
    return this.topAncestor(leafId);
  }
}
