"""对比检查：每扇前排/近排亮窗里字的中位亮度 / 底的中位亮度（读最后一次的成图和字蒙版）。"""
import importlib.util, sys
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("w3", "03_windows.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
front = m.front_buildings(); near = m.near_buildings()
m.set_visibility([w for b in near for w in b.wins], m.union_mask(front))
fw = [w for b in front for w in b.wins if w.lit]
for b in near:
    for w in b.wins:
        if w.lit and w.vis < 0.97: w.lit = False
nw = [w for b in near for w in b.wins if w.lit]
m.assign_texts(fw, nw)
img = np.asarray(Image.open("../03-窗.png")).astype(np.float32) / 255
tm = np.asarray(Image.open("../work/03_windows_text.png").convert("L").resize((2400, 3200), Image.BOX)).astype(np.float32) / 255
rows = []
for w in fw + nw:
    if not w.lit: continue
    X0, X1, Y0, Y1 = int(w.gx), int(w.gx + w.gw), int(w.gy), int(w.gy + w.gh)
    L = img[Y0:Y1, X0:X1].max(axis=2); t = tm[Y0:Y1, X0:X1]
    ink, bg = L[t > 0.6], L[t < 0.02]
    if len(ink) < 10: continue
    rows.append((np.median(ink) / max(np.median(bg), 1e-3), np.median(ink), np.median(bg), w.layer, int(w.gx), int(w.gy), int(w.gw), int(w.gh), m.ASKS[w.ask][1][:30]))
rows.sort()
for r in rows[:5]: print("low  %.1f ink %.2f bg %.2f L%d @%d,%d %dx%d %s" % r)
print("median ratio %.1f" % np.median([r[0] for r in rows]))
for r in rows:
    if "Bagaimana" in r[-1]: print("nasi %.1f ink %.2f bg %.2f L%d @%d,%d %dx%d %s" % r)
