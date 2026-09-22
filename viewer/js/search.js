// Game and territory search.
//
// The index is a separate lazy asset (`search.json`, 1.06 MB gzipped), fetched
// when the box is first focused rather than at load. Metadata is sharded by
// point-index block and only pulled on hover, so nothing on the client holds a
// name list until something asks for one.
//
// Cluster labels are not in that file. The manifest already carries every
// category with its label and bounding box and is loaded at startup, so label
// search runs against what is already in memory.

/**
 * Fold a string for matching: lowercase, strip accents, collapse punctuation.
 *
 * Accent stripping matters more than it looks on this catalog -- it contains
 * titles like "Pokémon", "Amnesia: Rebirth®" and "Ōkami", and a viewer typing
 * ASCII should find them.
 */
export function fold(s) {
  return String(s)
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

export class SearchIndex {
  constructor({ url = 'public/search.json' } = {}) {
    this.url = url;
    this.data = null;
    this.folded = null;
    this.loading = null;
    this.stats = { bytes: 0, ms: 0 };
  }

  get ready() { return this.data !== null; }

  load() {
    if (this.data) return Promise.resolve(this.data);
    if (this.loading) return this.loading;
    const t0 = performance.now();
    this.loading = fetch(this.url)
      .then(r => r.text())
      .then(text => {
        this.stats.bytes = text.length;
        const d = JSON.parse(text);
        // Fold once at load, not per keystroke: 56,129 normalise+lowercase
        // calls on every character typed is ~40ms of jank per key.
        this.folded = d.names.map(fold);
        this.data = d;
        this.stats.ms = Math.round(performance.now() - t0);
        return d;
      })
      .finally(() => { this.loading = null; });
    return this.loading;
  }

  /**
   * Rank: prefix matches first, then substring, ties broken by review count.
   *
   * "hollow" must surface Hollow Knight over obscure titles that merely
   * contain the word, which is what the prefix tier plus the popularity
   * tiebreak buys. A word-boundary tier sits between them so "knight" finds
   * "Hollow Knight" ahead of "Knightfall" only when neither starts with it.
   */
  games(query, limit = 8) {
    if (!this.data) return [];
    const q = fold(query);
    if (!q) return [];
    const out = [];
    const F = this.folded, D = this.data;
    for (let i = 0; i < F.length; i++) {
      const name = F[i];
      const at = name.indexOf(q);
      if (at < 0) continue;
      const tier = at === 0 ? 0 : (name[at - 1] === ' ' ? 1 : 2);
      out.push({ kind: 'game', i, tier, at, reviews: D.reviews[i] });
      // Bail out once there is plenty to rank. The catalog is 56k and a short
      // query like "a" matches most of it; ranking all of them is wasted work
      // when only 8 are shown.
      if (out.length > 4000) break;
    }
    out.sort((a, b) => a.tier - b.tier || b.reviews - a.reviews);
    return out.slice(0, limit).map(r => ({
      kind: 'game',
      index: r.i,
      appid: D.appids[r.i],
      name: D.names[r.i],
      x: D.x[r.i], y: D.y[r.i],
      reviews: D.reviews[r.i],
      year: D.years[r.i],
    }));
  }
}

/**
 * Match cluster labels from the manifest.
 *
 * Scored on the label's own terms, so typing "metroidvania" offers the
 * territory whose distinguishing term that is. Larger clusters win ties,
 * because a term that names 900 games is a more useful destination than one
 * naming 30.
 */
export function labels(H, query, limit = 4) {
  const q = fold(query);
  if (!q) return [];
  const out = [];
  for (const cat of H.byId.values()) {
    if (cat.parent === null) continue;
    const terms = (cat.terms && cat.terms.length)
      ? cat.terms : String(cat.label || '').split(' / ');
    let best = -1;
    for (let t = 0; t < terms.length; t++) {
      const f = fold(terms[t]);
      const at = f.indexOf(q);
      if (at < 0) continue;
      // earlier terms are more distinguishing; exact term beats a substring
      const tier = (f === q ? 0 : at === 0 ? 1 : 2) * 10 + t;
      if (best < 0 || tier < best) best = tier;
    }
    if (best < 0) continue;
    out.push({ kind: 'label', cat, tier: best, size: cat.size });
  }
  out.sort((a, b) => a.tier - b.tier || b.size - a.size);
  return out.slice(0, limit).map(r => ({
    kind: 'label',
    id: r.cat.id,
    name: r.cat.label,
    size: r.cat.size,
    confidence: r.cat.confidence,
    bbox: r.cat.bbox_core || r.cat.bbox,
    centroid: r.cat.centroid,
  }));
}
