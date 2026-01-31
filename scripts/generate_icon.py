#!/usr/bin/env python3
"""Generate a macOS .icns icon for Root Measure.

Draws a simple plant/root silhouette on a green gradient background.
Run once:  python scripts/generate_icon.py
Output:    assets/RootMeasure.icns
"""

import struct
import io
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent.parent
OUT = ROOT / "assets" / "RootMeasure.icns"

# --- drawing -----------------------------------------------------------

def draw_icon(size):
    """Return a PIL Image of the given size with a root icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size  # shorthand
    pad = s * 0.08
    # rounded-rect background (dark green gradient feel via solid)
    bg_color = (34, 120, 60)
    d.rounded_rectangle([pad, pad, s - pad, s - pad],
                        radius=s * 0.18, fill=bg_color)

    # stem (vertical line, top center going down ~45%)
    cx = s * 0.50
    stem_top = s * 0.16
    stem_bot = s * 0.50
    stem_w = s * 0.035
    d.line([(cx, stem_top), (cx, stem_bot)], fill="white", width=max(2, int(stem_w)))

    # two small leaves near top
    leaf_w = s * 0.12
    leaf_h = s * 0.06
    # left leaf
    lx, ly = cx - s * 0.01, stem_top + s * 0.10
    d.ellipse([lx - leaf_w, ly - leaf_h, lx, ly + leaf_h], fill=(180, 230, 180))
    # right leaf
    rx, ry = cx + s * 0.01, stem_top + s * 0.06
    d.ellipse([rx, ry - leaf_h, rx + leaf_w, ry + leaf_h], fill=(180, 230, 180))

    # main root (curving line from stem_bot downward)
    root_color = (230, 230, 220)
    root_w = max(2, int(s * 0.028))
    # primary root — slight curve to the left
    pts_main = [
        (cx, stem_bot),
        (cx - s * 0.02, s * 0.58),
        (cx - s * 0.04, s * 0.67),
        (cx - s * 0.03, s * 0.76),
        (cx - s * 0.02, s * 0.84),
    ]
    for i in range(len(pts_main) - 1):
        d.line([pts_main[i], pts_main[i + 1]], fill=root_color, width=root_w)

    # lateral root 1 — branches right
    thin_w = max(1, int(root_w * 0.6))
    branch1_start = pts_main[1]
    d.line([branch1_start, (cx + s * 0.12, s * 0.65)],
           fill=root_color, width=thin_w)
    d.line([(cx + s * 0.12, s * 0.65), (cx + s * 0.15, s * 0.72)],
           fill=root_color, width=thin_w)

    # lateral root 2 — branches left
    branch2_start = pts_main[2]
    d.line([branch2_start, (cx - s * 0.14, s * 0.74)],
           fill=root_color, width=thin_w)

    # lateral root 3 — small right branch lower
    branch3_start = pts_main[3]
    d.line([branch3_start, (cx + s * 0.08, s * 0.82)],
           fill=root_color, width=thin_w)

    return img


# --- .icns packing (subset of icon types) ------------------------------

ICNS_TYPES = {
    16:  b'icp4',   # 16x16
    32:  b'icp5',   # 32x32
    64:  b'icp6',   # 64x64 (actually 48 in old spec but 64 for PNG)
    128: b'ic07',   # 128x128
    256: b'ic08',   # 256x256
    512: b'ic09',   # 512x512
    1024: b'ic10',  # 1024x1024
}


def make_icns(sizes=None):
    """Build an .icns file from generated PNGs."""
    if sizes is None:
        sizes = [16, 32, 128, 256, 512, 1024]

    entries = []
    for sz in sizes:
        img = draw_icon(sz)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_data = buf.getvalue()
        ostype = ICNS_TYPES.get(sz)
        if ostype is None:
            continue
        # entry: 4-byte type + 4-byte length (includes header) + data
        entry = ostype + struct.pack(">I", 8 + len(png_data)) + png_data
        entries.append(entry)

    body = b"".join(entries)
    header = b"icns" + struct.pack(">I", 8 + len(body))
    return header + body


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = make_icns()
    OUT.write_bytes(data)
    print(f"Icon saved to {OUT}  ({len(data)} bytes)")
