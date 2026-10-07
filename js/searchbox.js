// The search box: input, dropdown, keyboard handling.
//
// Kept apart from app.js because it is the one piece of the viewer that is
// ordinary DOM UI rather than rendering, and mixing the two made app.js hard
// to read.

import { SearchIndex, labels } from './search.js';

const esc = (s) => String(s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export class SearchBox {
  /**
   * @param {object} opts
   * @param {Function} opts.onPickGame  ({index, x, y, ...}) => void
   * @param {Function} opts.onPickLabel ({bbox, centroid, ...}) => void
   */
  constructor({ input, list, hierarchy, onPickGame, onPickLabel }) {
    this.input = input;
    this.list = list;
    this.H = hierarchy;
    this.onPickGame = onPickGame;
    this.onPickLabel = onPickLabel;

    this.index = new SearchIndex();
    this.results = [];
    this.active = -1;
    this.debounce = null;

    // The index is 1.06 MB gzipped, so it loads on first focus rather than at
    // startup. Focusing is a reliable signal of intent and gives it a head
    // start before the first keystroke lands.
    input.addEventListener('focus', () => {
      this.index.load().then(() => { if (this.input.value) this.run(); });
    });

    input.addEventListener('input', () => {
      clearTimeout(this.debounce);
      this.debounce = setTimeout(() => this.run(), 60);
    });

    input.addEventListener('keydown', (e) => this.onKey(e));

    // `/` and ctrl/cmd+K focus the box, the way every other search field
    // behaves. `/` is ignored while typing in a field so it stays usable.
    window.addEventListener('keydown', (e) => {
      const typing = document.activeElement === input;
      if ((e.key === 'k' || e.key === 'K') && (e.ctrlKey || e.metaKey)) {
        e.preventDefault(); this.focus();
      } else if (e.key === '/' && !typing) {
        e.preventDefault(); this.focus();
      } else if (e.key === 'Escape' && typing) {
        if (this.list.hidden) { input.blur(); } else { this.close(); }
        e.stopPropagation();
      }
    });

    document.addEventListener('click', (e) => {
      if (!input.contains(e.target) && !list.contains(e.target)) this.close();
    });
  }

  focus() {
    this.input.focus();
    this.input.select();
  }

  close() {
    this.list.hidden = true;
    this.active = -1;
  }

  run() {
    const q = this.input.value.trim();
    if (!q) { this.close(); return; }
    if (!this.index.ready) { this.renderLoading(); return; }

    // Labels first: a territory is a coarser, usually more useful destination
    // than one game, and there are at most a few of them.
    const ls = labels(this.H, q, 3);
    const gs = this.index.games(q, 8 - Math.min(ls.length, 3));
    this.results = [...ls, ...gs];
    this.active = this.results.length ? 0 : -1;
    this.render();
  }

  renderLoading() {
    this.list.hidden = false;
    this.list.innerHTML = `<div class="loading">loading index…</div>`;
  }

  render() {
    if (!this.results.length) {
      this.list.hidden = false;
      this.list.innerHTML = `<div class="loading">no matches</div>`;
      return;
    }
    this.list.hidden = false;
    this.list.innerHTML = this.results.map((r, i) => {
      const on = i === this.active ? ' class="on"' : '';
      if (r.kind === 'label') {
        return `<div${on} data-i="${i}">
          <span class="kind">territory</span>
          <span class="nm lbl">${esc(r.name)}</span>
          <span class="sub">${r.size.toLocaleString()} games</span>
        </div>`;
      }
      return `<div${on} data-i="${i}">
        <span class="nm">${esc(r.name)}</span>
        <span class="sub">${r.year ?? '—'} · ${r.reviews.toLocaleString()}</span>
      </div>`;
    }).join('');

    for (const el of this.list.querySelectorAll('[data-i]')) {
      el.addEventListener('mouseenter', () => {
        this.active = +el.dataset.i; this.paint();
      });
      el.addEventListener('mousedown', (e) => {
        e.preventDefault();
        this.pick(+el.dataset.i);
      });
    }
  }

  paint() {
    for (const el of this.list.querySelectorAll('[data-i]')) {
      el.classList.toggle('on', +el.dataset.i === this.active);
    }
  }

  onKey(e) {
    if (this.list.hidden || !this.results.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      this.active = (this.active + 1) % this.results.length;
      this.paint();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      this.active = (this.active - 1 + this.results.length) % this.results.length;
      this.paint();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (this.active >= 0) this.pick(this.active);
    }
  }

  pick(i) {
    const r = this.results[i];
    if (!r) return;
    this.close();
    this.input.blur();
    if (r.kind === 'label') this.onPickLabel(r);
    else this.onPickGame(r);
  }
}
