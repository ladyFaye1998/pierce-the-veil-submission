"""
Generate Kaggle writeup media assets:

  assets/writeup_thumbnail_560x280.png   - the Card and Thumbnail image
                                            (Kaggle requires exactly 560 x 280)
  assets/writeup_card_1200x630.png       - higher-res OpenGraph-style card
                                            (used for social previews if needed)

Both are derived from assets/banner.png with a centered crop.

Author: Lady Faye
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def center_crop_resize(src: Path, target_w: int, target_h: int, dst: Path) -> None:
    im = Image.open(src).convert("RGB")
    src_w, src_h = im.size
    src_ratio = src_w / src_h
    dst_ratio = target_w / target_h
    if src_ratio > dst_ratio:
        new_h = src_h
        new_w = int(src_h * dst_ratio)
        left = (src_w - new_w) // 2
        top = 0
    else:
        new_w = src_w
        new_h = int(src_w / dst_ratio)
        left = 0
        top = (src_h - new_h) // 2
    im_crop = im.crop((left, top, left + new_w, top + new_h))
    im_out = im_crop.resize((target_w, target_h), Image.LANCZOS)
    im_out.save(dst, "PNG", optimize=True)
    print(f"  wrote {dst}  ({target_w}x{target_h})")


def main():
    src = ASSETS / "banner.png"
    if not src.exists():
        raise SystemExit(f"missing: {src}")
    print(f"source: {src} ({Image.open(src).size})")
    center_crop_resize(src, 560, 280, ASSETS / "writeup_thumbnail_560x280.png")
    center_crop_resize(src, 1200, 630, ASSETS / "writeup_card_1200x630.png")


if __name__ == "__main__":
    main()
