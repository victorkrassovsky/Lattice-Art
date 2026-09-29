#!/usr/bin/env python3
"""Convert an image into drawing instructions on a lattice-point grid.

Instructions are of two kinds:
  LINE  (x1,y1)-(x2,y2)   draw a segment between two lattice points (any angle/length, up to a cap)
  SHADE (x,y)             fill in the unit square whose lower-left corner is (x,y)

Coordinates use a standard Cartesian grid: (0,0) at the bottom-left,
x increasing right, y increasing up.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line


def load_grayscale(path):
    img = Image.open(path)
    img = img.convert("L")
    return img


def compute_cell_averages(gray_img, cols, rows):
    """Average brightness of each unit square, as a (rows, cols) array (row 0 = top)."""
    small = gray_img.resize((cols, rows), Image.BOX)
    return np.asarray(small, dtype=np.float64)


def detect_lines(gray_img, points_x, points_y, analysis_width, sigma, canny_low, canny_high,
                  hough_threshold, min_line_length, line_gap, max_degree, max_lines=None):
    """Find lattice-point-pair line segments that trace real edges in the image.

    This runs a standard, well-tested pipeline instead of a hand-rolled one:
    Canny edge detection (with hysteresis thresholding, so a weak edge pixel
    still counts if it's connected to a strong one -- much more robust than a
    single global magnitude cutoff) followed by a probabilistic Hough
    transform, which directly finds straight line segments in the edge map
    at any angle (not just a fixed set of directions).

    This runs at a fixed analysis resolution (analysis_width, aspect-matched
    height) independent of the lattice size, so a coarse lattice doesn't
    force edge-finding itself onto a tiny, blurry proxy image. Detected
    segments are then snapped to their nearest lattice points; degenerate
    (zero-length after snapping) and exact-duplicate segments are dropped,
    and a lattice point stops accepting more lines past max_degree so real
    edges read as clean chains rather than a tangle.
    """
    fine_w = max(points_x, analysis_width)
    fine_h = max(points_y, round(fine_w * gray_img.height / gray_img.width))
    fine = gray_img.resize((fine_w, fine_h), Image.LANCZOS)
    arr = np.asarray(fine, dtype=np.float64) / 255.0

    edges = canny(arr, sigma=sigma, low_threshold=canny_low, high_threshold=canny_high, use_quantiles=True)
    raw_segments = probabilistic_hough_line(
        edges, threshold=hough_threshold, line_length=min_line_length, line_gap=line_gap,
    )

    scale_x = (fine_w - 1) / (points_x - 1)
    scale_y = (fine_h - 1) / (points_y - 1)

    def to_lattice(x, y):
        col = min(max(round(x / scale_x), 0), points_x - 1)
        row = min(max(round(y / scale_y), 0), points_y - 1)
        return col, row

    def pt(col, row):
        return (int(col), points_y - 1 - int(row))

    scored = []  # (pixel_length, col0, row0, col1, row1)
    seen = set()
    for (x0, y0), (x1, y1) in raw_segments:
        c0, r0 = to_lattice(x0, y0)
        c1, r1 = to_lattice(x1, y1)
        if (c0, r0) == (c1, r1):
            continue
        key = frozenset(((c0, r0), (c1, r1)))
        if key in seen:
            continue
        seen.add(key)
        scored.append((math.hypot(x1 - x0, y1 - y0), c0, r0, c1, r1))

    # Longer, more confident strokes claim lattice points first.
    scored.sort(key=lambda s: s[0], reverse=True)

    degree = np.zeros((points_y, points_x), dtype=int)
    segments = []
    for _, c0, r0, c1, r1 in scored:
        if max_lines is not None and len(segments) >= max_lines:
            break
        if degree[r0, c0] >= max_degree or degree[r1, c1] >= max_degree:
            continue
        degree[r0, c0] += 1
        degree[r1, c1] += 1
        segments.append((pt(c0, r0), pt(c1, r1)))

    return segments


def compute_shaded_squares(cell_avgs, cols, rows, shade_threshold):
    """Squares darker than shade_threshold, identified by their lower-left lattice point."""
    squares = []
    for r in range(rows):
        for c in range(cols):
            if cell_avgs[r, c] < shade_threshold:
                squares.append((c, rows - 1 - r))
    return squares


def write_text_instructions(path, width, height, lines, squares):
    with open(path, "w") as f:
        f.write(f"# Lattice Art Instructions\n")
        f.write(f"# Grid: {width} x {height} points, (0,0) to ({width - 1},{height - 1})\n")
        f.write(f"# {len(lines)} lines, {len(squares)} shaded squares\n")
        for (x1, y1), (x2, y2) in lines:
            f.write(f"LINE ({x1},{y1})-({x2},{y2})\n")
        for (x, y) in squares:
            f.write(f"SHADE ({x},{y})\n")


def write_json_instructions(path, width, height, lines, squares):
    data = {
        "grid": {"width": width, "height": height},
        "lines": [{"from": list(a), "to": list(b)} for a, b in lines],
        "shaded_squares": [list(s) for s in squares],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def render_preview(path, width, height, lines, squares, scale, source_img=None):
    margin = scale
    canvas_w = (width - 1) * scale + 2 * margin
    canvas_h = (height - 1) * scale + 2 * margin

    def to_px(pt):
        x, y = pt
        return (margin + x * scale, margin + (height - 1 - y) * scale)

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img)

    for (x, y) in squares:
        top_left = to_px((x, y + 1))
        bottom_right = to_px((x + 1, y))
        draw.rectangle([top_left, bottom_right], fill=(40, 40, 40))

    for a, b in lines:
        draw.line([to_px(a), to_px(b)], fill=(0, 0, 0), width=max(1, scale // 10))

    dot_r = max(1, scale // 12)
    for r in range(height):
        for c in range(width):
            px, py = to_px((c, r))
            draw.ellipse([px - dot_r, py - dot_r, px + dot_r, py + dot_r], fill=(180, 180, 180))

    if source_img is not None:
        thumb = source_img.convert("RGB").copy()
        thumb.thumbnail((canvas_w, canvas_h))
        combined = Image.new("RGB", (canvas_w + thumb.width + margin, max(canvas_h, thumb.height)), "white")
        combined.paste(thumb, (0, 0))
        combined.paste(img, (thumb.width + margin, 0))
        combined.save(path)
    else:
        img.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="path to the source image")
    parser.add_argument("--points-x", type=int, default=40, help="number of lattice points across (default: 40)")
    parser.add_argument("--points-y", type=int, default=None, help="number of lattice points down (default: derived from image aspect ratio)")
    parser.add_argument("--shade-threshold", type=float, default=None, help="brightness (0-255) below which a square is shaded; overrides --shade-percentile if given (default: unset, use --shade-percentile instead)")
    parser.add_argument("--shade-percentile", type=float, default=12.0, help="shade roughly this percent of unit squares -- the darkest ones -- rather than using a fixed brightness cutoff; adapts to each image's tonal range. Lower = less shading, more reliance on lines (default: 12)")
    parser.add_argument("--analysis-width", type=int, default=600, help="width (px) of the internal image used to find edges, independent of --points-x -- keep this high even for a coarse lattice so detail-finding isn't done on an overly blurry proxy (default: 600)")
    parser.add_argument("--canny-sigma", type=float, default=2.0, help="Gaussian smoothing sigma (analysis-grid px) before edge detection -- higher suppresses more fine texture/noise (default: 2.0)")
    parser.add_argument("--canny-low", type=float, default=0.1, help="low hysteresis threshold for Canny edge detection, as a quantile (0-1) of gradient strength -- a weak edge pixel is kept if it connects to a strong one above --canny-high (default: 0.1)")
    parser.add_argument("--canny-high", type=float, default=0.2, help="high hysteresis threshold for Canny edge detection, as a quantile (0-1) of gradient strength -- a pixel above this is always a strong edge seed. Lower canny-low/canny-high = more, weaker edges detected = more lines (default: 0.2)")
    parser.add_argument("--hough-threshold", type=int, default=10, help="minimum Hough-transform vote count to accept a line -- lower finds more, fainter/shorter lines (default: 10)")
    parser.add_argument("--min-line-length", type=float, default=0.6, help="shortest line to detect, in lattice units -- shorter than this is discarded as insignificant at the output resolution (default: 0.6)")
    parser.add_argument("--line-gap", type=float, default=0.4, help="largest gap, in lattice units, to bridge when linking edge pixels into one line (default: 0.4)")
    parser.add_argument("--max-degree", type=int, default=3, help="max number of lines allowed to touch any one lattice point (default: 3)")
    parser.add_argument("--max-lines", type=int, default=None, help="cap the total number of lines drawn, keeping the longest/most confident first -- use this as your main 'how much detail' dial alongside --canny-low/--canny-high (default: unlimited)")
    parser.add_argument("--scale", type=int, default=20, help="pixels per lattice unit in the preview render (default: 20)")
    parser.add_argument("--out-dir", default=None, help="output directory (default: same directory as the image)")
    parser.add_argument("--out-prefix", default=None, help="output filename prefix (default: image basename)")
    parser.add_argument("--no-preview", action="store_true", help="skip rendering the preview PNG")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"error: no such file: {args.image}", file=sys.stderr)
        sys.exit(1)

    gray = load_grayscale(args.image)
    width_px, height_px = gray.size

    points_x = args.points_x
    points_y = args.points_y or max(2, round(points_x * height_px / width_px))
    if points_x < 2 or points_y < 2:
        print("error: grid must be at least 2x2 points", file=sys.stderr)
        sys.exit(1)

    cols, rows = points_x - 1, points_y - 1

    cell_avgs = compute_cell_averages(gray, cols, rows)
    # min-line-length/line-gap are expressed in lattice units for
    # intuitiveness; convert to analysis-grid pixels using the same scale
    # detect_lines derives internally.
    fine_w_est = max(points_x, args.analysis_width)
    unit_px = (fine_w_est - 1) / (points_x - 1)
    lines = detect_lines(
        gray, points_x, points_y, args.analysis_width,
        args.canny_sigma, args.canny_low, args.canny_high,
        args.hough_threshold, max(1, round(args.min_line_length * unit_px)), max(0, round(args.line_gap * unit_px)),
        args.max_degree, args.max_lines,
    )
    shade_threshold = args.shade_threshold
    if shade_threshold is None:
        shade_threshold = float(np.percentile(cell_avgs, args.shade_percentile))
    squares = compute_shaded_squares(cell_avgs, cols, rows, shade_threshold)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.image))
    prefix = args.out_prefix or os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, f"{prefix}_instructions.txt")
    json_path = os.path.join(out_dir, f"{prefix}_instructions.json")
    write_text_instructions(txt_path, points_x, points_y, lines, squares)
    write_json_instructions(json_path, points_x, points_y, lines, squares)
    print(f"wrote {txt_path}")
    print(f"wrote {json_path}")

    if not args.no_preview:
        preview_path = os.path.join(out_dir, f"{prefix}_preview.png")
        render_preview(preview_path, points_x, points_y, lines, squares, args.scale, source_img=Image.open(args.image))
        print(f"wrote {preview_path}")

    print(f"grid: {points_x} x {points_y} points, {len(lines)} lines, {len(squares)} shaded squares "
          f"(shade threshold {shade_threshold:.1f})")


if __name__ == "__main__":
    main()
