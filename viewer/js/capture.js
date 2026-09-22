// Drive a scripted zoom and post each rendered frame back to the server.
//
// The demo GIF has to be the real renderer -- points, labels, cross-fades and
// all -- so frames are composited from the live WebGL canvas plus the label
// overlay exactly as the viewer draws them. Anything reconstructed offline
// would be a picture of the thing rather than the thing.
//
// Usage from the console (or a driver script):
//   await window.__capture({ frames: 90, target: 'Hollow Knight' })

export function installCapture(atlas, { canvas, overlay, drawFrame }) {
  const composite = document.createElement('canvas');
  const cctx = composite.getContext('2d');

  async function post(name, blob) {
    await fetch(`/capture/${name}`, { method: 'POST', body: blob });
  }

  /**
   * The detail panel is DOM, so it is not in either canvas and cannot be
   * composited. Rather than leave the destination frame unexplained, the same
   * information is drawn onto the frame directly: what the map zoomed to, and
   * the game's true neighbours in the embedding.
   */
  function drawCaption(info, scale) {
    if (!info) return;
    const pad = Math.round(16 * scale * 2);
    const w = composite.width, h = composite.height;
    const lines = info.neighbours.slice(0, 5);
    const titleSize = Math.max(13, Math.round(22 * scale));
    const bodySize = Math.max(10, Math.round(14 * scale));
    const boxW = Math.round(w * 0.42);
    const boxH = pad * 1.9 + titleSize + bodySize * (lines.length + 3.4);
    const x = pad, y = h - boxH - pad;

    cctx.save();
    cctx.globalAlpha = info.alpha;
    cctx.fillStyle = 'rgba(13,17,23,0.86)';
    cctx.strokeStyle = 'rgba(48,54,61,1)';
    cctx.lineWidth = 1;
    cctx.beginPath();
    cctx.roundRect(x, y, boxW, boxH, 7);
    cctx.fill();
    cctx.stroke();

    let ty = y + pad * 0.85;
    cctx.textBaseline = 'top';
    cctx.fillStyle = '#f0f6fc';
    cctx.font = `600 ${titleSize}px ui-sans-serif, system-ui, sans-serif`;
    cctx.fillText(info.name, x + pad * 0.8, ty);
    ty += titleSize * 1.25;

    cctx.fillStyle = '#8b949e';
    cctx.font = `${bodySize}px ui-sans-serif, system-ui, sans-serif`;
    cctx.fillText(info.path, x + pad * 0.8, ty);
    ty += bodySize * 1.7;

    cctx.fillStyle = '#7ee787';
    cctx.fillText('nearest in the embedding', x + pad * 0.8, ty);
    ty += bodySize * 1.35;
    cctx.fillStyle = '#c9d1d9';
    for (const n of lines) {
      cctx.fillText('· ' + n, x + pad * 0.8, ty);
      ty += bodySize * 1.2;
    }
    cctx.restore();
  }

  function grab(scale, caption) {
    composite.width = Math.round(canvas.width * scale);
    composite.height = Math.round(canvas.height * scale);
    cctx.fillStyle = '#0d1117';
    cctx.fillRect(0, 0, composite.width, composite.height);
    cctx.drawImage(canvas, 0, 0, composite.width, composite.height);
    cctx.drawImage(overlay, 0, 0, composite.width, composite.height);
    drawCaption(caption, scale);
    return new Promise(res => composite.toBlob(res, 'image/png'));
  }

  /**
   * A single continuous zoom from the whole catalog to one game.
   *
   * Eased in log space, because zoom is multiplicative: a linear ramp on
   * `scale` crawls at the start and tears through the end, which reads as a
   * stutter rather than a glide. Position is interpolated with the same eased
   * parameter so the target drifts to centre as it is approached instead of
   * sliding first and then zooming.
   *
   * `zoom` is deliberately modest. An earlier attempt ended at 300x, which
   * puts the camera so close that the destination frame is a near-empty field
   * with one dot in it -- technically "down to one game" and useless to look
   * at. The end of the zoom should show the game *among its neighbours*, which
   * is the claim the map is making.
   */
  async function run({ frames = 80, target = 'Hollow Knight', zoom = 45,
                       scale = 0.55, settle = 2, captionFrom = 0.72 } = {}) {
    const { view, home, setFocus, neighbours, pathOf, metaOf, worldOf,
            findByName, refreshTiles } = atlas;

    const gid = await findByName(target);
    if (gid < 0) throw new Error(`no such game: ${target}`);

    // The target must be in the loaded tile set before its position is
    // readable, so the camera is parked on it once to pull those tiles in.
    const nb = await neighbours(gid);
    let where = worldOf(gid);
    if (!where) {
      const keep = { ...view };
      view.scale = home.scale * zoom;
      await refreshTiles(true);
      where = worldOf(gid);
      Object.assign(view, keep);
      await refreshTiles(true);
    }
    if (!where) throw new Error('target never appeared in a loaded tile');

    const from = { cx: home.cx, cy: home.cy, scale: home.scale };
    const to = { cx: where[0], cy: where[1], scale: home.scale * zoom };
    const ease = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

    const info = {
      name: metaOf(gid)?.name ?? target,
      path: pathOf(gid),
      neighbours: nb.map(n => metaOf(n)?.name ?? '…'),
      alpha: 0,
    };

    // focus early enough that the halo is visible during the approach, not
    // only after the camera stops
    const focusAt = Math.floor(frames * 0.55);
    const names = [];

    for (let f = 0; f < frames; f++) {
      const t = ease(f / (frames - 1));
      // geometric interpolation of scale; linear would stall then lurch
      view.scale = from.scale * Math.pow(to.scale / from.scale, t);
      view.cx = from.cx + (to.cx - from.cx) * t;
      view.cy = from.cy + (to.cy - from.cy) * t;
      if (f === focusAt) await setFocus(gid);
      await refreshTiles();

      info.alpha = Math.max(0, Math.min(1, (t - captionFrom) / (1 - captionFrom)));

      drawFrame();
      // Deliberately not requestAnimationFrame: rAF is throttled to ~1Hz in a
      // background tab, so a capture driven from a hidden pane would hang
      // forever. With preserveDrawingBuffer the frame is readable as soon as
      // the draw calls are issued, so yielding the task queue is enough.
      await new Promise(r => setTimeout(r, 0));
      const name = `f${String(f).padStart(4, '0')}.png`;
      await post(name, await grab(scale, info.alpha > 0.01 ? info : null));
      names.push(name);
    }

    // hold on the final frame so the GIF pauses on the destination
    info.alpha = 1;
    for (let h = 0; h < settle * 12; h++) {
      const name = `f${String(frames + h).padStart(4, '0')}.png`;
      await post(name, await grab(scale, info));
      names.push(name);
    }
    return { frames: names.length, target, zoom };
  }

  /**
   * One composited still: map, labels, and the hover card drawn in.
   *
   * The card is a DOM element, so it is not in either canvas. Rather than
   * screenshot the browser chrome, its content is drawn onto the composite so
   * the resulting file is exactly the map plus the card and nothing else.
   */
  async function snapshot({ name = 'hover.png', scale = 1, card = null } = {}) {
    // Draw first. requestAnimationFrame is throttled in a background tab,
    // so the canvas may hold nothing at all by the time a snapshot is
    // requested -- which composites as a black rectangle.
    drawFrame();
    composite.width = Math.round(canvas.width * scale);
    composite.height = Math.round(canvas.height * scale);
    cctx.fillStyle = '#0d1117';
    cctx.fillRect(0, 0, composite.width, composite.height);
    cctx.drawImage(canvas, 0, 0, composite.width, composite.height);
    cctx.drawImage(overlay, 0, 0, composite.width, composite.height);

    if (card) {
      const { x, y, w, img, m, tint } = card;
      // Height from the content actually drawn, not the DOM element's box:
      // the two differ and the difference shows as dead space.
      const ih0 = Math.round(w * 0.466);
      const h = ih0 + 108;
      const rr = (a, b, c, d, r) => {
        cctx.beginPath(); cctx.roundRect(a, b, c, d, r);
      };
      cctx.save();
      cctx.shadowColor = 'rgba(0,0,0,.55)'; cctx.shadowBlur = 24;
      cctx.shadowOffsetY = 8;
      cctx.fillStyle = 'rgba(22,27,34,.97)';
      rr(x, y, w, h, 8); cctx.fill();
      cctx.restore();

      cctx.save();
      rr(x, y, w, h, 8); cctx.clip();
      const ih = ih0;
      if (img && img !== 'missing') {
        cctx.drawImage(img, x, y, w, ih);
      } else {
        cctx.fillStyle = tint; cctx.fillRect(x, y, w, ih);
        cctx.fillStyle = 'rgba(255,255,255,.62)';
        cctx.font = '600 26px ui-sans-serif, system-ui, sans-serif';
        cctx.textAlign = 'center'; cctx.textBaseline = 'middle';
        cctx.fillText(m.name.slice(0, 2).toUpperCase(), x + w / 2, y + ih / 2);
        cctx.textAlign = 'left';
      }
      cctx.restore();

      let ty = y + ih + 16;
      cctx.textBaseline = 'alphabetic';
      cctx.fillStyle = '#e6edf3';
      cctx.font = '600 15px ui-sans-serif, system-ui, sans-serif';
      cctx.fillText(m.name, x + 12, ty); ty += 20;

      cctx.font = '11px ui-sans-serif, system-ui, sans-serif';
      let tx = x + 12;
      for (const tag of m.tags.slice(0, 4)) {
        const tw = cctx.measureText(tag).width + 10;
        if (tx + tw > x + w - 12) break;
        cctx.fillStyle = '#21262d';
        rr(tx, ty - 10, tw, 15, 3); cctx.fill();
        cctx.fillStyle = '#adbac7';
        cctx.fillText(tag, tx + 5, ty + 1);
        tx += tw + 4;
      }
      ty += 22;

      const good = m.score >= 0.8, mixed = m.score >= 0.6;
      cctx.fillStyle = good ? '#7ee787' : mixed ? '#e3b341' : '#ff7b72';
      cctx.font = '600 12px ui-sans-serif, system-ui, sans-serif';
      cctx.fillText(m.scoreWord, x + 12, ty);
      cctx.fillStyle = '#6e7681';
      cctx.font = '12px ui-sans-serif, system-ui, sans-serif';
      const rt = `${m.reviews.toLocaleString()} reviews`;
      cctx.fillText(rt, x + w - 12 - cctx.measureText(rt).width, ty);
      ty += 18;

      cctx.fillStyle = '#6e7681';
      cctx.fillText(String(m.year ?? '—'), x + 12, ty);
      cctx.fillStyle = '#e6edf3';
      cctx.font = '600 12px ui-sans-serif, system-ui, sans-serif';
      const pr = m.priceText;
      cctx.fillText(pr, x + w - 12 - cctx.measureText(pr).width, ty);
      ty += 20;

      cctx.strokeStyle = 'rgba(48,54,61,1)'; cctx.lineWidth = 1;
      cctx.beginPath(); cctx.moveTo(x + 12, ty - 10);
      cctx.lineTo(x + w - 12, ty - 10); cctx.stroke();
      cctx.fillStyle = '#6e7681';
      cctx.font = '11px ui-sans-serif, system-ui, sans-serif';
      cctx.fillText('click to open on Steam ↗', x + 12, ty + 3);
    }

    const blob = await new Promise(res => composite.toBlob(res, 'image/png'));
    await post(name, blob);
    return { name, bytes: blob.size, size: [composite.width, composite.height] };
  }

  window.__capture = run;
  window.__snapshot = snapshot;
  return run;
}
