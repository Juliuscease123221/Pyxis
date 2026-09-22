// Lazy game metadata, sharded by point-index block.
//
// The first version loaded a single 5.62 MB `meta.json` before the first
// frame, for data only needed when something is hovered. It is now split into
// 14 shards of ~0.43 MB, fetched on demand and cached.
//
// Sharding is by point index rather than by tile because a point appears in
// several tiles -- the redundancy that makes tiles independently renderable --
// so tile-keyed metadata would ship the same record once per zoom level.

export class MetaStore {
  constructor({ shardSize, base = 'public/meta' }) {
    this.shardSize = shardSize;
    this.base = base;
    this.shards = new Map();     // shard index -> record arrays
    this.inflight = new Map();
    this.stats = { requests: 0, bytes: 0 };
  }

  shardOf(i) { return Math.floor(i / this.shardSize); }

  has(i) { return this.shards.has(this.shardOf(i)); }

  async load(i) {
    const s = this.shardOf(i);
    if (this.shards.has(s)) return this.shards.get(s);
    if (this.inflight.has(s)) return this.inflight.get(s);

    const p = (async () => {
      this.stats.requests++;
      const res = await fetch(`${this.base}/${String(s).padStart(4, '0')}.json`);
      const text = await res.text();
      this.stats.bytes += text.length;
      const data = JSON.parse(text);
      this.shards.set(s, data);
      return data;
    })().finally(() => this.inflight.delete(s));

    this.inflight.set(s, p);
    return p;
  }

  /** Synchronous accessor; returns null if the shard is not loaded yet. */
  get(i) {
    const s = this.shards.get(this.shardOf(i));
    if (!s) return null;
    const k = i - s.from;
    return {
      name: s.names[k],
      appid: s.appids[k],
      tags: s.tags[k] || [],
      reviews: s.reviews[k] || 0,
    };
  }

  /** Load whatever shards these indices need, then return them. */
  async ensure(indices) {
    const need = new Set(indices.map(i => this.shardOf(i)));
    await Promise.all([...need].map(s => this.load(s * this.shardSize)));
  }
}
