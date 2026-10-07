// The hover card: a preview built from metadata plus the Steam header image.
//
// Not an embedded store page. Steam sends `X-Frame-Options: SAMEORIGIN`, so an
// iframe renders nothing; the card is assembled here and the store page opens
// in a new tab on click instead.
//
// ### Why images are fetched only on hover
//
// 56,129 header images at ~44 KB each is roughly 2.4 GB. Preloading even the
// visible tile's worth would be tens of megabytes for something the viewer may
// never look at. Images are therefore requested when a point is actually
// hovered, after a debounce, and cached in memory by appid.
//
// ### The appid
//
// A tile point record carries `id:u32`, which in this build is a **row index**
// -- the tile format takes an opaque integer and the library never learns what
// it means. The real Steam appid comes from the metadata shard. Both the image
// URL and the store link depend on it, so `meta.get(row).appid` is the single
// source for both.

const HEADER_URL = (appid) =>
  `https://cdn.cloudflare.steamstatic.com/steam/apps/${appid}/header.jpg`;

export const STORE_URL = (appid) =>
  `https://store.steampowered.com/app/${appid}/`;

/**
 * In-memory image cache keyed by appid.
 *
 * Stores the outcome, not just the image: a 404 is cached as `'missing'` so a
 * delisted game is not re-requested every time the cursor crosses it. Plenty
 * of games in this catalog have no header any more.
 */
export class ImageCache {
  constructor(limit = 300) {
    this.limit = limit;
    this.map = new Map();        // appid -> HTMLImageElement | 'missing' | Promise
    this.stats = { requests: 0, hits: 0, missing: 0 };
  }

  peek(appid) {
    const v = this.map.get(appid);
    return (v && typeof v.then === 'function') ? null : v ?? null;
  }

  load(appid) {
    const have = this.map.get(appid);
    if (have) {
      if (typeof have.then !== 'function') this.stats.hits++;
      return Promise.resolve(have);
    }
    this.stats.requests++;
    const p = new Promise((resolve) => {
      const img = new Image();
      // The CDN sends Access-Control-Allow-Origin: *, so requesting
      // anonymously keeps a canvas that draws this image untainted --
      // which is what lets the docs snapshot be exported to a file.
      img.crossOrigin = 'anonymous';
      img.decoding = 'async';
      img.referrerPolicy = 'no-referrer';
      img.onload = () => resolve(img);
      img.onerror = () => { this.stats.missing++; resolve('missing'); };
      img.src = HEADER_URL(appid);
    }).then((res) => {
      this.map.set(appid, res);
      while (this.map.size > this.limit) {
        this.map.delete(this.map.keys().next().value);
      }
      return res;
    });
    this.map.set(appid, p);
    return p;
  }
}

function scoreClass(score) {
  if (score >= 0.8) return 'good';
  if (score >= 0.6) return 'mixed';
  return 'poor';
}

function scoreWord(score, reviews) {
  if (!reviews) return 'No reviews';
  if (score >= 0.95) return 'Overwhelmingly positive';
  if (score >= 0.8) return 'Very positive';
  if (score >= 0.7) return 'Mostly positive';
  if (score >= 0.4) return 'Mixed';
  if (score >= 0.2) return 'Mostly negative';
  return 'Overwhelmingly negative';
}

const esc = (s) => String(s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export class HoverCard {
  constructor(el, images) {
    this.el = el;
    this.images = images;
    this.appid = null;
  }

  hide() {
    this.el.hidden = true;
    this.appid = null;
  }

  /**
   * Render the card for `m` and place it near (px, py) in CSS pixels.
   *
   * `avoid` is the set of screen positions that must not be covered -- the
   * haloed neighbours. The card flips to whichever side leaves them visible,
   * and flips again at the viewport edges so it is never clipped.
   */
  show(m, colour, px, py, avoid = []) {
    const score = m.score ?? 0;
    const price = (m.price === null || m.price === undefined)
      ? null
      : (m.price === 0 ? 'Free' : `$${Number(m.price).toFixed(2)}`);

    if (this.appid !== m.appid) {
      this.appid = m.appid;
      const tint = `rgb(${colour.map(c => Math.round(c * 255 * 0.55)).join(',')})`;
      // background-COLOR, not the `background` shorthand.
      //
      // The shorthand resets every background-* longhand, including
      // background-size and background-position, and an inline style beats the
      // stylesheet. Writing `background: <tint>` here silently reverted
      // `background-size: cover` to `auto` and the position to `0% 0%`, so the
      // header drew at its natural 460px inside a 267px box -- 1.73x, anchored
      // top-left, right and bottom cropped away. Steam headers are composed
      // with the title centred, so that crop destroys them.
      //
      // The image is now a real <img> at width:100% in a container locked to
      // 460/215, which cannot crop at all.
      this.el.innerHTML = `
        <div class="shot" style="background-color:${tint}">
          <img class="hdr" alt="" hidden>
          <div class="ph">${esc(m.name.slice(0, 2).toUpperCase())}</div>
        </div>
        <div class="body">
          <div class="title">${esc(m.name)}</div>
          <div class="tagrow">${m.tags.slice(0, 4).map(t =>
            `<span>${esc(t)}</span>`).join('')}</div>
          <div class="meta">
            <span class="score ${scoreClass(score)}">${scoreWord(score, m.reviews)}</span>
            <span class="dim">${m.reviews.toLocaleString()} reviews</span>
          </div>
          <div class="meta">
            <span class="dim">${m.year ?? '—'}</span>
            <span class="price">${price ?? '—'}</span>
          </div>
          <div class="hint">click to select · ctrl-click opens Steam ↗</div>
        </div>`;

      // Image arrives asynchronously; the coloured placeholder is what shows
      // until it does, and what stays if it 404s.
      const shot = this.el.querySelector('.shot');
      const el = shot.querySelector('img.hdr');
      const attach = (res) => {
        if (this.appid !== m.appid || res === 'missing' || !res) return;
        el.src = res.src;
        el.hidden = false;
        shot.classList.add('loaded');
      };
      const cached = this.images.peek(m.appid);
      if (cached && cached !== 'missing') attach(cached);
      else if (cached !== 'missing') this.images.load(m.appid).then(attach);
    }

    this.el.hidden = false;
    this.place(px, py, avoid);
  }

  place(px, py, avoid) {
    const el = this.el;
    const w = el.offsetWidth || 268;
    const h = el.offsetHeight || 190;
    const pad = 14;
    const vw = window.innerWidth, vh = window.innerHeight;

    // Prefer right-then-below; flip on each axis if it would overflow, and
    // flip horizontally if the default side would sit on top of the haloed
    // neighbours -- occluding them defeats the point of showing them.
    let x = px + pad;
    let y = py + pad;
    if (x + w > vw - 8) x = px - w - pad;
    if (y + h > vh - 8) y = py - h - pad;

    if (avoid.length) {
      const covered = (cx) => avoid.filter(([ax, ay]) =>
        ax > cx && ax < cx + w && ay > y && ay < y + h).length;
      const right = px + pad, left = px - w - pad;
      if (left > 8 && covered(right) > covered(left)) x = left;
    }

    el.style.left = `${Math.max(8, Math.min(x, vw - w - 8))}px`;
    el.style.top = `${Math.max(8, Math.min(y, vh - h - 8))}px`;
  }
}
