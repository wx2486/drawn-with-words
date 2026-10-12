"""成图 → 发布用 JPEG（原尺寸，X 单图 ≤5 MB）+ 展示页用的缩图。

发布目录在仓库外（作者本地的 AIGC/图片合集/发布/），没有就只出展示页的图。
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]            # AIGC/表达
AIGC = ROOT.parent
SRC = ROOT / "图"
PUB = AIGC / "图片合集" / "发布" / "2026-10-12"
PAGE = ROOT / "页面" / "img"
ITEMS = [("01-海", "1-sea"), ("02-我", "2-self"), ("03-窗", "3-windows"), ("04-螺", "4-spiral"), ("05-图", "5-map")]

HAS_PUB = PUB.parent.exists()
if HAS_PUB:
    PUB.mkdir(exist_ok=True)
PAGE.mkdir(parents=True, exist_ok=True)
for zh, en in ITEMS:
    im = Image.open(SRC / f"{zh}.png").convert("RGB")
    if HAS_PUB:
        im.save(PUB / f"{en}.jpg", quality=92, subsampling=0, optimize=True)
    im.save(PAGE / f"{en}-full.jpg", quality=90, subsampling=0, optimize=True)
    small = im.resize((1200, 1600), Image.LANCZOS)
    small.save(PAGE / f"{en}.jpg", quality=86, subsampling=0, optimize=True)
    print(f"{en}: page {(PAGE / f'{en}.jpg').stat().st_size / 1e3:.0f} KB")
