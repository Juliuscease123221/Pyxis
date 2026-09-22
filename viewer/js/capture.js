// Drive a scripted zoom and post each rendered frame back to the server.
//
// The demo GIF has to be the real renderer -- points, labels, cross-fades and
// all -- so frames are composited from the live WebGL canvas plus the label
// overlay exactly as the viewer draws them. Anything reconstructed offline
// would be a picture of the thing rather than the thing.
//
// Usage from the console (or a driver script):
//   await window.__capture({ frames: 90, target: 'Hollow Knight' })

export function installCapture(atlas, { canvas, overlay, meta, drawFrame }) {
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
    const i = meta.names.indexOf(target);
    if (i < 0) throw new Error(`no such game: ${target}`);

    const { view, home, P, setFocus, neighbours, pathOf } = atlas;
    const from = { cx: home.cx, cy: home.cy, scale: home.scale };
    const to = { cx: P.x[i], cy: P.y[i], scale: home.scale * zoom };
    const ease = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

    const nb = neighbours(i);
    const info = {
      name: meta.names[i],
      path: pathOf(i),
      neighbours: nb.map(n => meta.names[n]),
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
      if (f === focusAt) setFocus(i);

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

  window.__capture = run;
  return run;
}
