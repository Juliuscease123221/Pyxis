"""Assemble captured frames into the demo GIF.

Frames come from the live renderer via `viewer/js/capture.js`, so the GIF shows
the real thing -- the same shaders, the same label placement, the same
cross-fades -- rather than a reconstruction.

GIF is a 256-colour format and the map is a dark field with many similar hues,
so a naive per-frame palette produces visible banding and a palette that
flickers between frames. Both are avoided by quantising every frame against
one shared adaptive palette built from a sample of the sequence.

Usage:
    python -m viewer.make_gif --out docs/demo.gif --fps 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPTURES = ROOT / "viewer" / "captures"


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=str(CAPTURES))
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=880)
    ap.add_argument("--colors", type=int, default=200)
    ap.add_argument("--every", type=int, default=1,
                    help="keep every Nth frame. During a zoom almost every"
                         " pixel changes between frames, so GIF inter-frame"
                         " compression buys nothing and file size is close to"
                         " linear in frame count -- this is the main lever.")
    ap.add_argument("--dither", action="store_true",
                    help="Floyd-Steinberg dithering. Off by default: the map is"
                         " mostly flat dark field, and dithering it scatters"
                         " single-pixel noise that GIF's run-length compression"
                         " cannot pack, roughly doubling the file for no"
                         " visible gain on this content.")
    args = ap.parse_args()

    paths = sorted(Path(args.frames).glob("f*.png"))[::max(1, args.every)]
    if not paths:
        print(f"no frames in {args.frames}")
        return 1

    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.width != args.width:
            h = round(im.height * args.width / im.width)
            im = im.resize((args.width, h), Image.LANCZOS)
        frames.append(im)

    # One palette for the whole sequence. Built from frames spread across the
    # zoom so it covers both the wide view (many territory hues at once) and
    # the close view (few hues, but subtle) -- sampling only the start would
    # leave the destination banded.
    step = max(1, len(frames) // 12)
    sample = frames[::step]
    stack = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, im in enumerate(sample):
        stack.paste(im, (0, i * im.height))
    palette = stack.quantize(colors=args.colors, method=Image.MEDIANCUT)

    dither = Image.FLOYDSTEINBERG if args.dither else Image.NONE
    quantised = [f.quantize(palette=palette, dither=dither) for f in frames]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(
        out, save_all=True, append_images=quantised[1:],
        duration=round(1000 / args.fps), loop=0, optimize=True, disposal=2,
    )
    mb = out.stat().st_size / 1e6
    print(f"{len(frames)} frames -> {out}  ({frames[0].width}x{frames[0].height}, "
          f"{args.fps:g} fps, {mb:.2f} MB)")
    if mb > 12:
        print("  note: large for a README; reduce --width or --fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
