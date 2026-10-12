"""看图用：整图缩略 + 指定区域 100% 裁切。python preview.py <png> [x y w h]..."""
import sys
from pathlib import Path
from PIL import Image
p = Path(sys.argv[1])
im = Image.open(p)
out = p.parent / "work"
out.mkdir(exist_ok=True)
im.resize((900, int(900 * im.height / im.width)), Image.LANCZOS).save(out / f"{p.stem}_small.png")
args = list(map(int, sys.argv[2:]))
for i in range(0, len(args), 4):
    x, y, w, h = args[i:i + 4]
    im.crop((x, y, x + w, y + h)).save(out / f"{p.stem}_crop{i // 4}.png")
print("ok")
