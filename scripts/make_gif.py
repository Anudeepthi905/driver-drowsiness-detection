"""
Convert a short demo clip into an optimised GIF for the README.

Usage:
    python scripts/make_gif.py demo.mp4
    python scripts/make_gif.py demo.mp4 --start 1.5 --duration 9

Output: docs/demo.gif (at most 10 s, about 12 fps, 640 px wide, 8 MB or less).

Requires ffmpeg on your PATH (Windows: `winget install Gyan.FFmpeg`,
macOS: `brew install ffmpeg`, Linux: `sudo apt install ffmpeg`),
or pass its location with --ffmpeg.

How it keeps the GIF small and sharp: a GIF can only hold 256 colours, so
ffmpeg first builds a palette of the colours this particular clip uses
(palettegen), then maps every frame onto it (paletteuse). If the result is
still over the size limit, the script retries with fewer colours and then a
smaller width until it fits.
"""

import argparse
import os
import shutil
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(REPO_DIR, "docs", "demo.gif")
MAX_DURATION = 10.0

# (width, colours) tried in order until the GIF fits under the size limit.
ATTEMPTS = [(640, 256), (640, 128), (640, 64), (560, 64), (480, 64)]


def build_command(ffmpeg, src, dst, start, duration, fps, width, colours):
    # min(width, iw): never upscale a clip that is already narrower than `width`.
    filters = (
        f"fps={fps},scale='min({width}\\,iw)':-1:flags=lanczos,split[a][b];"
        f"[a]palettegen=max_colors={colours}:stats_mode=diff[p];"
        f"[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    )
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-t", str(duration), "-i", src,
            "-vf", filters, "-loop", "0", dst]


def main():
    parser = argparse.ArgumentParser(description="Turn a demo clip into docs/demo.gif.")
    parser.add_argument("input", help="video clip, e.g. demo.mp4")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="GIF path (default: docs/demo.gif)")
    parser.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start of the clip")
    parser.add_argument("--duration", type=float, default=MAX_DURATION, help="length in seconds (max 10)")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate (default 12)")
    parser.add_argument("--max-mb", type=float, default=8.0, help="size limit in MB (default 8)")
    parser.add_argument("--ffmpeg", default=None, help="path to ffmpeg if it is not on your PATH")
    args = parser.parse_args()

    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("[ERROR] ffmpeg not found. Install it (see the top of this file) or pass --ffmpeg PATH.")
    if not os.path.isfile(args.input):
        sys.exit(f"[ERROR] Input clip not found: {args.input}")

    duration = min(args.duration, MAX_DURATION)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    limit = args.max_mb * 1024 * 1024

    for width, colours in ATTEMPTS:
        cmd = build_command(ffmpeg, args.input, args.output, args.start, duration,
                            args.fps, width, colours)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"[ERROR] ffmpeg failed:\n{result.stderr.strip()}")
        size = os.path.getsize(args.output)
        print(f"  width {width}px, {colours} colours -> {size / 1024 / 1024:.2f} MB")
        if size <= limit:
            print(f"[OK] Saved {args.output}")
            return

    sys.exit(f"[ERROR] Still over {args.max_mb} MB at the smallest setting. "
             "Try a shorter --duration.")


if __name__ == "__main__":
    main()
