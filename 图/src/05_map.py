"""图 5「图」：一张用字画的旧地图——我知道的世界，大致。

做法：陆地是几组椭圆经噪声扭曲后的场，取 0 等值线当海岸线；外面的水纹是「全部陆地」的海上距离场的偏移等值线
（所以相邻陆地的水纹会合并，不会交叉）。所有线（海岸、水纹、河、边框、罗盘、比例尺、小路）都是 texts_map 里的字，
一个词一个词地沿路径摆：每个词单独一个 textPath，字宽先用 headless Chrome 实测并缓存（work/05_measure.json）；
闭合的海岸线按周长微调字号，让首尾正好接上；朝左走的词换到反向路径上，免得倒着读；进雾、碰到标签就按概率把词拿掉，
所以海岸线进雾以后会变淡、断成虚线。纸、水渍、霉斑、折痕、雾都在 numpy 里做；墨、朱红、雾里的字是三张 Chrome 蒙版。

    python -X utf8 05_map.py
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure as skm

sys.path.insert(0, str(Path(__file__).parent))
from common import *          # noqa
from texts_map import TITLE, SUBTITLE, LANDS, FOG, INTERIOR, PHANTOM, PATH, SEAS, COMPASS, SCALE

SEED = 11
G = 2                                   # 场的网格：半分辨率
GH, GW = H // G, W // G
FF = "'Noto Serif SC', Cambria, 'Malgun Gothic', 'Nirmala UI', 'Times New Roman', serif"
FF_IT = "Cambria, Georgia, 'Noto Serif SC', serif"

# 地图框（内边）
FX0, FY0, FX1, FY1 = 150, 150, W - 150, H - 150


# ================================================================ 陆地

def ellipse_field(X, Y, cx, cy, rx, ry, ang):
    a = np.deg2rad(ang)
    c, s = np.cos(a), np.sin(a)
    u = (X - cx) * c + (Y - cy) * s
    v = -(X - cx) * s + (Y - cy) * c
    d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
    return (1 - d) * min(rx, ry)


def smax(fs, k):
    fs = np.stack(fs)
    m = fs.max(0)
    return m + k * np.log(np.exp((fs - m) / k).sum(0))


# 每块地：若干椭圆 (cx, cy, rx, ry, 角度[, -1 表示挖掉])，噪声强度 amp，扭曲强度 warp
REGIONS = {
    "lang": dict(idx=0, amp=1.0, warp=1.0, blobs=[
        (1260, 820, 470, 200, -6), (1660, 760, 230, 160, 14), (980, 960, 200, 150, 28),
        (1430, 990, 190, 110, -20), (790, 1085, 170, 85, -32), (1200, 640, 120, 60, 0, -1)]),
    "code": dict(idx=1, amp=0.45, warp=0.45, blobs=[
        (1180, 2160, 160, 115, 42), (1290, 2310, 125, 88, 52), (1385, 2440, 100, 74, 40),
        (1470, 2550, 76, 58, 25), (1550, 2615, 52, 40, 10)]),
    "math": dict(idx=2, amp=0.45, warp=0.25, k=4, blobs=[
        (1790, 2430, 92, 66, 20), (1985, 2350, 50, 38, -10), (2080, 2560, 78, 56, 35),
        (1860, 2650, 46, 34, 0), (2010, 2790, 66, 46, -25), (1720, 2820, 36, 28, 10),
        (2170, 2390, 40, 30, 0)]),
    "hist": dict(idx=3, amp=1.0, warp=0.9, blobs=[
        (790, 1640, 420, 520, 8), (540, 1430, 230, 270, 0), (1040, 1900, 290, 260, 18),
        (660, 2050, 290, 210, -12), (1070, 1380, 200, 180, 0), (420, 2170, 110, 64, -30),
        (230, 1720, 100, 80, 0, -1), (930, 2300, 140, 80, 0, -1)]),
    "lit": dict(idx=4, amp=0.7, warp=0.6, blobs=[
        (440, 860, 190, 120, -18), (360, 950, 105, 85, 0)]),
    "sci": dict(idx=5, amp=1.0, warp=1.0, blobs=[
        (1990, 1580, 270, 470, 6), (2230, 1320, 240, 320, 0), (1900, 2010, 180, 200, 0),
        (1800, 1350, 120, 90, 0, -1)]),
}
LAKE = (800, 1640, 125, 85, 14)
PHANTOM_BLOBS = [(1010, 2770, 88, 64, -12), (1068, 2806, 62, 50, 25), (962, 2740, 46, 40, 0)]


def land_fields():
    """返回 (F 全陆地场, label 每格属于哪块地, phantom 场)，都在 G 网格上，单位 px（正为陆地）。"""
    yy, xx = np.mgrid[0:GH, 0:GW].astype(np.float32)
    X, Y = xx * G + G / 2, yy * G + G / 2
    wx = fbm(GH, GW, SEED + 1, octaves=4, base=380 / G) * 60
    wy = fbm(GH, GW, SEED + 2, octaves=4, base=380 / G) * 60
    n1 = fbm(GH, GW, SEED + 3, octaves=3, base=320 / G)
    n2 = fbm(GH, GW, SEED + 4, octaves=3, base=120 / G)
    n3 = fbm(GH, GW, SEED + 5, octaves=2, base=64 / G)
    noise = 70 * n1 + 30 * n2 + 6 * n3
    fields, names = [], []
    for name, r in sorted(REGIONS.items(), key=lambda kv: kv[1]["idx"]):
        Xw, Yw = X + r["warp"] * wx, Y + r["warp"] * wy
        pos = [ellipse_field(Xw, Yw, *b[:5]) for b in r["blobs"] if len(b) == 5]
        neg = [ellipse_field(Xw, Yw, *b[:5]) for b in r["blobs"] if len(b) == 6]
        f = smax(pos, r.get("k", 45))
        for nb in neg:
            f = -smax([-f, nb], 25)
        f = f + r["amp"] * noise
        if name == "hist":
            lake = ellipse_field(Xw, Yw, *LAKE) + 0.5 * noise
            f = -smax([-f, lake], 12)
        fields.append(f)
        names.append(name)
    fields = np.stack(fields)
    F = smax(list(fields), 8)
    label = fields.argmax(0)
    # 去掉太小的岛和湖（雾湖除外）
    land = F > 0
    lab, n = ndimage.label(land)
    sizes = ndimage.sum(land, lab, range(1, n + 1)) * G * G
    for i, sz in enumerate(sizes, 1):
        if sz < 16000:
            comp = lab == i
            if REGIONS["math"]["idx"] != np.bincount(label[comp]).argmax() or sz < 900:
                F[comp] = np.minimum(F[comp], -1.0)
    sea = F <= 0
    lab, n = ndimage.label(sea)
    sizes = ndimage.sum(sea, lab, range(1, n + 1)) * G * G
    for i, sz in enumerate(sizes, 1):
        if sz < 30000:
            F[lab == i] = np.maximum(F[lab == i], 1.0)
    # 幻岛：噪声小，形状笃定
    ph = smax([ellipse_field(X + 0.2 * wx, Y + 0.2 * wy, *b) for b in PHANTOM_BLOBS], 30) + 0.25 * (30 * n2 + 6 * n3)
    return F.astype(np.float32), label, ph.astype(np.float32), names


def preview_land():
    F, label, ph, names = land_fields()
    land = (F > 0) | (ph > 0)
    inner = land[FY0 // G:FY1 // G, FX0 // G:FX1 // G]
    print(f"陆地占比 {inner.mean() * 100:.1f}%")
    cols = np.array([[200, 170, 120], [150, 190, 140], [130, 160, 200], [210, 140, 120], [190, 140, 200],
                     [140, 200, 200]], np.uint8)
    img = np.full((GH, GW, 3), 235, np.uint8)
    img[F > 0] = cols[label[F > 0]]
    img[ph > 0] = (60, 60, 60)
    for c in skm.find_contours(F, 0) + skm.find_contours(ph, 0):
        cv2.polylines(img, [np.round(c[:, ::-1] * 4).astype(np.int32)], True, (40, 30, 20), 1, cv2.LINE_AA, shift=2)
    cv2.rectangle(img, (FX0 // G, FY0 // G), (FX1 // G, FY1 // G), (0, 0, 0), 1)
    for v in range(0, W, 100):
        cv2.line(img, (v // G, 0), (v // G, GH), (180, 180, 180) if v % 500 else (90, 90, 200), 1)
    for v in range(0, H, 100):
        cv2.line(img, (0, v // G), (GW, v // G), (180, 180, 180) if v % 500 else (90, 90, 200), 1)
    from PIL import Image
    Image.fromarray(img).save(WORK / "05_land.png")
    print("saved", WORK / "05_land.png")


# ================================================================ 量字宽（Chrome 实测，缓存）

MEASURE_CACHE = WORK / "05_measure.json"


def _mkey(text, weight, style, family):
    return f"{weight}|{style}|{family}|{text}"


def measure_widths(specs):
    """specs: [(text, weight, style, family)] → {key: 字号 100 时的宽度}。"""
    cache = json.loads(MEASURE_CACHE.read_text(encoding="utf-8")) if MEASURE_CACHE.exists() else {}
    need = sorted({sp for sp in specs if _mkey(*sp) not in cache})
    if need:
        html = """<!doctype html><html><head><meta charset="utf-8"></head><body>
<svg id="s" xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg><pre id="out"></pre>
<script>
const specs = %s; const svg = document.getElementById('s'); const res = [];
for (const [t, w, st, f] of specs) {
  const e = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  e.setAttribute('font-size', '100'); e.setAttribute('font-weight', w);
  e.setAttribute('font-style', st); e.setAttribute('font-family', f);
  e.setAttribute('xml:space', 'preserve'); e.textContent = t; svg.appendChild(e);
  res.push(e.getComputedTextLength());
}
document.getElementById('out').textContent = 'RES' + JSON.stringify(res) + 'END';
</script></body></html>""" % json.dumps([list(sp) for sp in need], ensure_ascii=False)
        hp = WORK / "05_measure.html"
        hp.write_text(html, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="chrome-ud-") as ud:
            out = subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={ud}",
                                  "--virtual-time-budget=10000", "--dump-dom", hp.as_uri()],
                                 capture_output=True, timeout=300).stdout.decode("utf-8")
        res = json.loads(out[out.index("RES") + 3:out.index("END")])
        for sp, w in zip(need, res):
            cache[_mkey(*sp)] = w
        MEASURE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"  量了 {len(need)} 段字宽")
    return {_mkey(*sp): cache[_mkey(*sp)] for sp in specs}


class Widths:
    """字宽表：缺的先按估算返回并记下，整张图排一遍后一次量完再重排。"""
    def __init__(self):
        self.table, self.missing = {}, set()
        if MEASURE_CACHE.exists():
            self.table = json.loads(MEASURE_CACHE.read_text(encoding="utf-8"))

    def want(self, text, weight=400, style="normal", family=FF):
        if _mkey(text, str(weight), style, family) not in self.table:
            self.missing.add((text, str(weight), style, family))

    def run(self):
        if self.missing:
            self.table.update(measure_widths(list(self.missing)))
            self.missing = set()

    def __call__(self, text, size, weight=400, style="normal", family=FF):
        k = _mkey(text, str(weight), style, family)
        if k not in self.table:
            self.missing.add((text, str(weight), style, family))
            return sum(1.0 if ord(c) > 0x2E80 else 0.52 for c in text) * size
        return self.table[k] * size / 100.0


WID = Widths()


# ================================================================ 曲线与沿线排字

def resample(pts, step, closed):
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / step), 6)
    t = np.linspace(0, s[-1], n + 1)
    if closed:
        t = t[:-1]
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1)


def smooth_line(pts, sigma, closed, step=2.0):
    pts = resample(pts, step, closed)
    if sigma > 0:
        pts = ndimage.gaussian_filter1d(pts, sigma / step, axis=0, mode="wrap" if closed else "nearest")
    return resample(pts, step, closed)


class Line:
    _n = 0

    def __init__(self, pts, closed):
        self.pts = np.asarray(pts, np.float64)
        self.closed = closed
        q = np.vstack([self.pts, self.pts[:1]]) if closed else self.pts
        self.q = q
        seg = np.hypot(*np.diff(q, axis=0).T)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.L = float(self.s[-1])
        Line._n += 1
        self.id = f"p{Line._n}"

    def at(self, s):
        s = np.clip(s, 0, self.L)
        return np.stack([np.interp(s, self.s, self.q[:, 0]), np.interp(s, self.s, self.q[:, 1])], -1)

    def defs(self, need_rev=True):
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in self.q)
        out = f'<path id="{self.id}" d="{d}"/>'
        if need_rev:
            r = self.q[::-1]
            out += f'<path id="{self.id}r" d="M ' + " L ".join(f"{x:.1f} {y:.1f}" for x, y in r) + '"/>'
        return out


def sample(arr, xy):
    """在全分辨率数组上按 (x, y) 取值（最近邻）。"""
    xy = np.atleast_2d(xy)
    x = np.clip(np.round(xy[:, 0]).astype(int), 0, arr.shape[1] - 1)
    y = np.clip(np.round(xy[:, 1]).astype(int), 0, arr.shape[0] - 1)
    return arr[y, x]


class Stream:
    """一块地的词流：每次取下一个词。"""
    def __init__(self, items, rng):
        self.items = items
        self.k = int(rng.integers(0, len(items)))

    def next(self):
        t = self.items[self.k % len(self.items)]
        self.k += 1
        return t


def lay_line(line, pick, size, weight=400, gap_em=1.05, mode="closed", opacity=1.0, drop=None,
             rng=None, sep=True, family=FF, style="normal", flip=True, max_bend=0.84, start=0.0):
    """沿 line 一个词一个词地摆，每个词一个 textPath。
    mode：closed 闭合线（微调字号让首尾正好接上）；fit 开放线两端对齐（调空隙）；free 从 start 摆到放不下。
    size 可以是 f(s/L) → 字号（只用于 free）。drop(xy) → 这个词留下的概率（雾、标签、画框）。"""
    sizef = size if callable(size) else (lambda u, _s=size: _s)
    L = line.L
    seq, s = [], (start if mode == "free" else 0.0)
    while True:
        sz = sizef(min(s / L, 1.0))
        t = pick(s)
        w = WID(t, sz, weight, style, family)
        if mode == "closed":
            if s + w > L + 0.5 * w:
                break
        elif s + w > L - 0.1 * sz:
            break
        seq.append([t, w, sz, gap_em * sz])
        s += w + gap_em * sz
        if len(seq) > 5000:
            break
    if not seq:
        return []
    if mode == "closed":
        tot = sum(it[1] + it[3] for it in seq)
        scale = L / tot
        if scale > 1.12:
            t = pick(s)
            sz = sizef(1.0)
            w = WID(t, sz, weight, style, family)
            tot2 = tot + w + gap_em * sz
            if abs(L / tot2 - 1) < abs(scale - 1):
                seq.append([t, w, sz, gap_em * sz])
                scale = L / tot2
        scale = float(np.clip(scale, 0.75, 1.3))
        for it in seq:
            it[1] *= scale
            it[2] *= scale
            it[3] *= scale
        s = 0.0
    elif mode == "fit":
        while len(seq) > 1 and (L - sum(it[1] for it in seq)) / (len(seq) - 1) < 0.45 * gap_em * seq[0][2]:
            seq.pop()
        if len(seq) > 1:
            g = (L - sum(it[1] for it in seq)) / (len(seq) - 1)
            for it in seq:
                it[3] = g
            s = 0.0
        else:
            s = max(0.0, (L - seq[0][1]) / 2)
    else:
        s = start
    out = []
    state = None
    for t, w, sz, g in seq:
        if s + w > L + 0.5:
            break
        a, b = line.at(np.array([s, s + w]))
        chord = float(np.hypot(*(b - a)))
        dx = (b[0] - a[0]) / max(chord, 1e-6)
        if state is None:
            state = dx < -0.05
        elif dx < -0.3:
            state = True
        elif dx > 0.3:
            state = False
        ok = chord >= max_bend * w
        keep = 1.0
        if ok and drop is not None:
            keep = drop(line.at(np.linspace(s, s + w, max(3, int(w / 10) + 2))))
        if ok and (keep >= 1.0 or (keep > 0 and rng is not None and rng.random() < keep)):
            href, off = (line.id + "r", L - s - w) if (flip and state) else (line.id, s)
            at = [f'font-size="{sz:.2f}"']
            if str(weight) != "400":
                at.append(f'font-weight="{weight}"')
            if style != "normal":
                at.append(f'font-style="{style}"')
            if family != FF:
                at.append(f'font-family="{family}"')
            if opacity < 1:
                at.append(f'fill-opacity="{opacity:.2f}"')
            out.append(f'<text {" ".join(at)}><textPath href="#{href}" startOffset="{off:.1f}">{esc(t)}</textPath></text>')
            if sep and g > 0.45 * sz:
                mid = s + w + 0.5 * g
                if mid < L - 1:
                    pm = line.at(np.array([mid]))
                    k2 = 1.0 if drop is None else drop(pm)
                    if k2 >= 1.0 or (k2 > 0 and rng is not None and rng.random() < k2):
                        op = f' fill-opacity="{opacity:.2f}"' if opacity < 1 else ""
                        out.append(f'<circle cx="{pm[0][0]:.1f}" cy="{pm[0][1]:.1f}" r="{0.075 * sz:.2f}"{op}/>')
        s += w + g
    return out


def text_el(x, y, t, size, weight=400, style="normal", ls=0.0, anchor="middle", family=FF, rot=0.0, op=1.0,
            baseline=None):
    at = [f'font-size="{size:.2f}"']
    if str(weight) != "400":
        at.append(f'font-weight="{weight}"')
    if style != "normal":
        at.append(f'font-style="{style}"')
    if family != FF:
        at.append(f'font-family="{family}"')
    if ls:
        at.append(f'letter-spacing="{ls:.2f}"')
    if anchor != "start":
        at.append(f'text-anchor="{anchor}"')
    if op < 1:
        at.append(f'fill-opacity="{op:.2f}"')
    if baseline:
        at.append(f'style="dominant-baseline:{baseline}"')
    lx = ls / 2 if (anchor == "middle" and ls) else 0.0
    return (f'<g transform="translate({x:.1f} {y:.1f}) rotate({rot:.2f})">'
            f'<text x="{lx:.2f}" y="0" {" ".join(at)}>{esc(t)}</text></g>')


def keep_rect(mask, cx, cy, w, h, rot=0.0, pad=8.0):
    a = np.deg2rad(rot)
    c, s = np.cos(a), np.sin(a)
    hw, hh = w / 2 + pad, h / 2 + pad
    pts = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
    pts = np.stack([cx + pts[:, 0] * c - pts[:, 1] * s, cy + pts[:, 0] * s + pts[:, 1] * c], 1)
    cv2.fillPoly(mask, [np.round(pts * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)


def keep_poly(mask, pts, width):
    cv2.polylines(mask, [np.round(np.asarray(pts) * 4).astype(np.int32)], False, 1,
                  thickness=max(1, int(round(width))), lineType=cv2.LINE_8, shift=2)


def label(layer, x, y, t, size, weight=400, style="normal", ls=0.0, family=FF, rot=0.0, op=1.0, keep=(), pad=10.0):
    w = WID(t, size, weight, style, family) + ls * max(len(t) - 1, 0)
    layer.append(text_el(x, y, t, size, weight, style, ls, "middle", family, rot, op))
    for m in keep:
        keep_rect(m, x, y, w, size * 1.15, rot, pad)
    return w


def catmull(ctrl, step=3.0):
    P = np.asarray(ctrl, np.float64)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(np.hypot(*(p2 - p1)) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def meander(pts, amp, wl, seed):
    """沿法线加一点低频弯曲，让手画的线不那么规整。"""
    pts = resample(pts, 2.0, False)
    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    s = np.arange(len(pts)) * 2.0
    r = np.random.default_rng(seed)
    off = sum(amp / (k + 1) * np.sin(2 * np.pi * s / (wl / (k + 1)) + r.uniform(0, 6.3)) for k in range(3))
    off *= np.minimum(1, np.minimum(s, s[-1] - s) / 60.0)
    return pts + nrm * off[:, None]


# ================================================================ 几何：海岸、水纹、雾

PH_IDX = 6
REGION_ITEMS = [[t.strip() for t in L[2].split(" · ") if t.strip()] for L in LANDS] + [list(PHANTOM)]
RINGS = [(26, 11.0, 0.80), (50, 9.2, 0.62), (77, 7.6, 0.46), (108, 6.3, 0.32)]   # (离岸距离, 字号, 浓度)


def up(a, interp=cv2.INTER_LINEAR):
    return cv2.resize(a.astype(np.float32), (W, H), interpolation=interp)


def contours(field_g, level, min_len=40, sigma=5.0, scale=G):
    out = []
    for c in skm.find_contours(field_g, level):
        pts = np.stack([c[:, 1] * scale + scale / 2, c[:, 0] * scale + scale / 2], 1)
        closed = np.hypot(*(pts[0] - pts[-1])) < 1e-3
        if closed:
            pts = pts[:-1]
        if len(pts) < 4:
            continue
        pts = smooth_line(pts, sigma, closed)
        L = Line(pts, closed)
        if L.L >= min_len:
            out.append(L)
    return out


def orient(line, field, outward_is_low=True, thresh=0.0):
    """让每条线左手边（字头朝向）是海 / 离岸更远的一侧。"""
    p = line.pts
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
    left = p + 6 * np.stack([t[:, 1], -t[:, 0]], 1)
    v = sample(field, left)
    bad = (v > thresh).mean() if outward_is_low else (v < thresh).mean()
    if bad > 0.5:
        return Line(p[::-1].copy(), line.closed)
    return line


def fog_field(lake_c):
    """雾的浓度：东边一整片、三个角、大陆中间的湖；边缘不规则，带横向拉长的雾丝。只在图框里面。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n1 = fbm(H, W, SEED + 30, octaves=4, base=520)
    n2 = fbm(H, W, SEED + 31, octaves=6, base=260)
    wisp = cv2.resize(fbm(H, W // 3, SEED + 32, octaves=5, base=110), (W, H), interpolation=cv2.INTER_CUBIC)
    east = smoothstep(1700, 2310, xx + 240 * n1 + 80 * wisp - 60 * np.cos(yy / 520.0))

    def corner(cx, cy, r, k=1.0):
        d = np.hypot(xx - cx, yy - cy) * (1 + 0.28 * n1) + 40 * wisp
        return k * smoothstep(r, r * 0.25, d)
    fog = np.maximum.reduce([east, corner(W, 0, 900), corner(W, H, 950), corner(0, H, 430, 0.7)])
    fog = np.clip(fog * (0.80 + 0.30 * smoothstep(-0.8, 0.8, n2) + 0.14 * wisp), 0, 1)
    dl = np.hypot((xx - lake_c[0]) / 1.15, yy - lake_c[1]) * (1 + 0.30 * n2) + 30 * wisp
    fog = np.maximum(fog, 0.84 * smoothstep(275, 30, dl))
    inside = (smoothstep(FX0 - 2, FX0 + 26, xx) * smoothstep(FX1 + 2, FX1 - 26, xx)
              * smoothstep(FY0 - 2, FY0 + 26, yy) * smoothstep(FY1 + 2, FY1 - 26, yy))
    return np.clip(fog * inside, 0, 1).astype(np.float32), wisp.astype(np.float32)


class Geo:
    pass


def build_geo():
    g = Geo()
    F, label_g, ph, names = land_fields()
    Ff, phf = up(F), up(ph)
    landf = (Ff > 0) | (phf > 0)
    labf = up(label_g, cv2.INTER_NEAREST).astype(np.int8)
    labf[phf > 0] = PH_IDX
    sea = ~landf
    dist, (iy, ix) = ndimage.distance_transform_edt(sea, return_indices=True)
    g.dist = dist.astype(np.float32)
    g.region = labf[iy, ix]
    g.land = landf
    g.din = ndimage.distance_transform_edt(landf).astype(np.float32)
    g.Ff = Ff
    sl, n = ndimage.label(sea)
    sizes = ndimage.sum(sea, sl, range(1, n + 1))
    big = 1 + int(np.argmax(sizes))
    g.lake = (sl != big) & sea & (sl > 0)
    ys, xs = np.nonzero(g.lake)
    g.lake_c = (float(xs.mean()), float(ys.mean())) if len(xs) else (LAKE[0], LAKE[1])
    g.fog, g.wisp = fog_field(g.lake_c)
    g.coasts = [orient(L, Ff) for L in contours(F, 0.0, min_len=60, sigma=5)]
    g.phantom = [orient(L, phf) for L in contours(ph, 0.0, min_len=60, sigma=4)]
    g.rings, g.lake_rings = [], []
    dist_g = cv2.resize(g.dist, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    for k, (d, size, op) in enumerate(RINGS):
        for L in contours(dist_g, d, min_len=80, sigma=4 + 0.06 * d, scale=2):
            L = orient(L, g.dist, outward_is_low=False, thresh=d)
            if g.lake[int(np.clip(L.pts[0, 1], 0, H - 1)), int(np.clip(L.pts[0, 0], 0, W - 1))]:
                if k <= 1:
                    g.lake_rings.append((k, L))             # 雾湖里只留里面两圈，隐约透出来
                continue
            g.rings.append((k, L))
    print(f"  海岸 {len(g.coasts)} 条，幻岛 {len(g.phantom)} 条，水纹 {len(g.rings)} 条，湖心 {g.lake_c[0]:.0f},{g.lake_c[1]:.0f}")
    return g


def in_frame(xy, pad=14):
    return bool(((xy[:, 0] > FX0 + pad) & (xy[:, 0] < FX1 - pad) & (xy[:, 1] > FY0 + pad)
                 & (xy[:, 1] < FY1 - pad)).all())


def make_drop(g, mask, fog_lo=0.22, fog_hi=0.80, frame=True):
    def drop(xy):
        if frame and not in_frame(xy):
            return 0.0
        if mask is not None and sample(mask, xy).any():
            return 0.0
        f = float(sample(g.fog, xy).mean())
        return float(1 - smoothstep(fog_lo, fog_hi, f))
    return drop


# ================================================================ 版面（坐标都是成图像素）

CART = (195, 195, 1005, 560)                     # 标题框
COMPASS_C, COMPASS_R = (425, 2812), 156.0
SCALE_BAR = (1180, 1660, 2988)
PATH_CTRL = [   # 从历史大陆东岸的港口出海，缓 S 形穿过常识之海，在自然科学登岸，往东走进雾里
    (1200, 1545), (1252, 1546), (1320, 1552), (1392, 1563), (1462, 1524), (1512, 1452), (1565, 1388),
    (1632, 1349), (1710, 1336), (1790, 1332), (1862, 1330), (1910, 1333), (1966, 1343), (2026, 1352),
    (2094, 1333), (2166, 1306), (2262, 1290)]
RIVERS = [
    ([(700, 1290), (712, 1370), (742, 1450), (785, 1540)], 3),
    ([(968, 1800), (1012, 1880), (996, 1980), (1040, 2080), (1010, 2200)], 3),
    ([(1240, 715), (1226, 820), (1192, 930), (1150, 1070)], 0),
    ([(1985, 1748), (1900, 1730), (1815, 1716), (1710, 1700)], 5),
]
RIDGES = [   # (山脊控制点, 最大字号, 带宽的一半)
    ([(430, 1272), (560, 1242), (700, 1262), (840, 1236), (985, 1270)], 30, 44),
    ([(1065, 1960), (1110, 1850), (1150, 1750), (1190, 1650)], 26, 30),
    ([(1020, 700), (1180, 664), (1350, 646), (1520, 668), (1690, 646)], 30, 42),
    ([(2125, 1040), (2062, 1150), (2008, 1262)], 26, 36),
    ([(2060, 1600), (2025, 1655), (1990, 1705)], 22, 28),
    ([(330, 822), (420, 796), (508, 814)], 22, 26),
]
FORESTS = [(520, 2165, 120, 58, 60), (395, 1565, 62, 88, 40), (1785, 782, 92, 60, 50), (1858, 2118, 66, 56, 35),
           (1160, 2235, 50, 34, 16)]
LAND_NAMES = [   # (区域, x, y, 中文字号, 中文字距, 英文字号, 角度)
    (4, 422, 912, 40, 14, 15, 0), (0, 1500, 860, 58, 30, 19, 0), (3, 770, 2008, 64, 36, 20, 0),
    (5, 1928, 1905, 50, 22, 17, 0), (1, 1338, 2398, 36, 10, 14, 50), (2, 1888, 2552, 34, 16, 13, 0),
]
FOG_SPOTS = [(1, 1965, 330), (0, 2108, 1168), (3, 2080, 2168), (2, 2010, 2985)]   # (FOG 序号, x, y)


class Layers:
    def __init__(self):
        self.ink, self.red, self.fogt = [], [], []
        self.ink_defs, self.red_defs = [], []


def build_frame(Ly, rng):
    st = Stream([TITLE[0], TITLE[1]], rng)
    for inset, size, wgt, op, gap in [(74, 7.2, 700, 1.0, 1.1), (88, 4.6, 400, 0.9, 1.1), (113, 12.5, 400, 0.62, 1.8),
                                      (138, 5.2, 600, 0.95, 1.1)]:
        x0, y0, x1, y1 = inset, inset, W - inset, H - inset
        m = size * 0.9
        sides = [[(x0 + m, y0), (x1 - m, y0)], [(x1, y0 + m), (x1, y1 - m)],
                 [(x1 - m, y1), (x0 + m, y1)], [(x0, y1 - m), (x0, y0 + m)]]
        for sd in sides:
            L = Line(sd, False)
            Ly.ink += lay_line(L, lambda s: st.next(), size, weight=wgt, gap_em=gap, mode="fit", opacity=op, rng=rng)
            Ly.ink_defs.append(L.defs())


def build_cartouche(Ly, keep, rng):
    x0, y0, x1, y1 = CART
    st = Stream([TITLE[0], TITLE[1]], rng)
    for ins, size, wgt in [(0, 5.8, 700), (9, 4.2, 400)]:
        a0, b0, a1, b1 = x0 + ins, y0 + ins, x1 - ins, y1 - ins
        m = size * 0.9
        for sd in [[(a0 + m, b0), (a1 - m, b0)], [(a1, b0 + m), (a1, b1 - m)], [(a1 - m, b1), (a0 + m, b1)],
                   [(a0, b1 - m), (a0, b0 + m)]]:
            L = Line(sd, False)
            Ly.ink += lay_line(L, lambda s: st.next(), size, weight=wgt, gap_em=1.1, mode="fit", rng=rng)
            Ly.ink_defs.append(L.defs())
    cx = (x0 + x1) / 2
    label(Ly.red, cx, 318, TITLE[0], 74, weight=600, ls=16)
    label(Ly.red, cx, 402, TITLE[1], 24, weight=500, ls=9)
    w1 = WID(SUBTITLE[0], 30, 500)
    w2 = WID(SUBTITLE[1], 25, 400, "italic", FF_IT)
    gapw = 34
    tot = w1 + gapw + w2
    label(Ly.ink, cx - tot / 2 + w1 / 2, 476, SUBTITLE[0], 30, weight=500)
    Ly.ink.append(f'<circle cx="{cx - tot / 2 + w1 + gapw / 2:.1f}" cy="476" r="2.2"/>')
    label(Ly.ink, cx + tot / 2 - w2 / 2, 474, SUBTITLE[1], 25, style="italic", family=FF_IT)
    keep_rect(keep, cx, (y0 + y1) / 2, x1 - x0, y1 - y0, 0, 34)


def seg_intersect(p, d, a, b):
    """射线 p + t·d 与线段 ab 的交点参数 t（无交点返回 None）。"""
    e = b - a
    den = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(den) < 1e-9:
        return None
    r = a - p
    t = (r[0] * (-e[1]) - r[1] * (-e[0])) / den
    u = (d[0] * r[1] - d[1] * r[0]) / den
    return t if 0 <= u <= 1 else None


def build_compass(Ly, keep, rng):
    C = np.array(COMPASS_C, np.float64)
    R = COMPASS_R
    lab = {0: COMPASS[0], 90: COMPASS[1], 180: COMPASS[2], 270: COMPASS[3]}

    def u(b):
        a = np.deg2rad(b)
        return np.array([np.sin(a), -np.cos(a)])
    r_in = 0.30 * R
    star = np.zeros((H, W), np.uint8)
    polys = []
    for b in (45, 135, 225, 315, 0, 90, 180, 270):
        Lr = R if b % 90 == 0 else 0.62 * R
        tip = C + u(b) * Lr
        il, ir = C + u(b - 22.5) * r_in, C + u(b + 22.5) * r_in
        items = [lab[b]] if b in lab else [lab[(b - 45) % 360], lab[(b + 45) % 360]]
        st = Stream(items, rng)
        size = 5.2 if b % 90 == 0 else 4.5
        for a_, b_ in [(il, tip), (ir, tip), (C + u(b) * 6, tip)]:
            L = Line([a_, b_], False)
            Ly.ink += lay_line(L, lambda s: st.next(), size, weight=600, gap_em=0.8, mode="fit", rng=rng, max_bend=0.5)
            Ly.ink_defs.append(L.defs())
        # 左半边用细字排成影线
        d = u(b)
        nrm = (il - C) - d * np.dot(il - C, d)
        nrm /= np.linalg.norm(nrm)
        w_in = np.dot(il - C, nrm)
        for o in np.arange(3.0, w_in - 1.2, 4.2):
            p = C + nrm * o
            t0 = seg_intersect(p, d, C, il)
            t1 = seg_intersect(p, d, il, tip)
            if t0 is None or t1 is None or t1 - t0 < 12:
                continue
            L = Line([p + d * (t0 + 2), p + d * (t1 - 2)], False)
            Ly.ink += lay_line(L, lambda s: st.next(), 3.7, weight=500, gap_em=0.6, mode="fit", rng=rng, sep=False,
                               max_bend=0.5)
            Ly.ink_defs.append(L.defs())
        polys.append(np.array([C, il, tip, ir]))
    for pg in polys:
        cv2.fillPoly(star, [np.round(pg * 4).astype(np.int32)], 1, shift=2)
    star = cv2.dilate(star, np.ones((9, 9), np.uint8))
    st = Stream(list(COMPASS), rng)
    for r, size in [(0.80 * R, 5.2), (0.87 * R, 4.3), (1.13 * R, 5.6)]:
        a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        L = Line(np.stack([C[0] + r * np.cos(a), C[1] + r * np.sin(a)], 1), True)
        L = Line(L.pts[::-1].copy(), True)
        drop = lambda xy, _m=star: 0.0 if sample(_m, xy).any() else 1.0
        Ly.ink += lay_line(L, lambda s: st.next(), size, weight=500, gap_em=1.0, mode="closed", drop=drop, rng=rng)
        Ly.ink_defs.append(L.defs())
    for b, t in lab.items():
        p = C + u(b) * (R + 40)
        label(Ly.ink, p[0], p[1], t, 24, weight=600)
    cv2.circle(keep, (int(C[0]), int(C[1])), int(R + 70), 1, -1)


def build_scale(Ly, keep, rng):
    x0, x1, y = SCALE_BAR
    st = Stream([SCALE[0], SCALE[1]], rng)
    for yy_, size, wgt in [(y - 8, 4.4, 600), (y + 8, 4.4, 600)]:
        L = Line([(x0, yy_), (x1, yy_)], False)
        Ly.ink += lay_line(L, lambda s: st.next(), size, weight=wgt, gap_em=0.9, mode="fit", rng=rng)
        Ly.ink_defs.append(L.defs())
    bounds = [0, 0.07, 0.19, 0.255, 0.46, 0.535, 0.78, 1.0]
    for k in range(len(bounds) - 1):
        a, b = x0 + bounds[k] * (x1 - x0), x0 + bounds[k + 1] * (x1 - x0)
        if k % 2 == 0:
            for yy_ in (y - 3.6, y + 0.2, y + 4.0):
                L = Line([(a + 2, yy_), (b - 2, yy_)], False)
                Ly.ink += lay_line(L, lambda s: st.next(), 3.3, weight=500, gap_em=0.5, mode="fit", rng=rng, sep=False)
                Ly.ink_defs.append(L.defs())
    cx = (x0 + x1) / 2
    label(Ly.ink, cx, y - 50, SCALE[0], 21, weight=500, ls=3)
    label(Ly.ink, cx, y + 34, SCALE[1], 16, style="italic", family=FF_IT, ls=1.5)
    keep_rect(keep, cx, y - 8, x1 - x0, 110, 0, 22)


def build_labels(Ly, g, keep, occ, rng):
    # 陆地名：中文拉开字距，英文小号大写在下
    for idx, x, y, zs, zls, es, rot in LAND_NAMES:
        zh, en = LANDS[idx][0], LANDS[idx][1].upper()
        a = np.deg2rad(rot)
        if rot:                     # 窄的半岛：英文接在中文后面，沿同一条轴
            wz = WID(zh, zs, 500) + zls * (len(zh) - 1)
            we = WID(en, es, 500) + 0.45 * es * (len(en) - 1)
            ax = np.array([np.cos(a), np.sin(a)])
            c0 = np.array([x, y]) - ax * (we + 26) / 2
            label(Ly.ink, *c0, zh, zs, weight=500, ls=zls, rot=rot, keep=(keep, occ), pad=12)
            c1 = c0 + ax * (wz / 2 + 26 + we / 2)
            label(Ly.ink, *c1, en, es, weight=500, ls=0.45 * es, rot=rot, keep=(keep, occ), pad=10)
            continue
        off = np.array([-np.sin(a), np.cos(a)]) * (0.5 * zs + 0.9 * es)
        label(Ly.ink, x, y, zh, zs, weight=500, ls=zls, rot=rot, keep=(keep, occ), pad=12)
        label(Ly.ink, x + off[0], y + off[1], en, es, weight=500, ls=0.45 * es, rot=rot, keep=(keep, occ), pad=10)
    # 海名：拉开字距的斜体
    zh, en = SEAS[0]
    for k, ch in enumerate(zh):
        label(Ly.ink, 1540, 1648 + k * 92, ch, 44, style="italic", op=0.9, keep=(keep,), pad=12)
    label(Ly.ink, 1474, 1786, en, 17, style="italic", family=FF_IT, ls=4, rot=90, op=0.9, keep=(keep,), pad=8)
    zh, en = SEAS[1]
    label(Ly.ink, 560, 2470, zh, 46, style="italic", ls=34, rot=-3, op=0.9, keep=(keep,), pad=12)
    label(Ly.ink, 562, 2526, en, 18, style="italic", family=FF_IT, ls=5, rot=-3, op=0.9, keep=(keep,), pad=8)
    # 雾里的话（画在雾上面）
    for k, x, y in FOG_SPOTS:
        zh, en = FOG[k]
        label(Ly.fogt, x, y, zh, 31, weight=500, style="italic", ls=5, keep=(keep, occ), pad=12)
        label(Ly.fogt, x, y + 36, en, 17, style="italic", family=FF_IT, ls=1.5, keep=(keep, occ), pad=8)
    x, y = g.lake_c
    label(Ly.fogt, x, y - 8, INTERIOR[0], 27, weight=500, style="italic", ls=4, keep=(keep, occ), pad=10)
    label(Ly.fogt, x, y + 24, INTERIOR[1], 15, style="italic", family=FF_IT, keep=(keep, occ), pad=8)
    # 幻岛的标签（朱红）
    px = 1006
    label(Ly.red, px, 2462, PHANTOM[0], 31, weight=600, ls=7, keep=(keep, occ), pad=10)
    label(Ly.red, px, 2499, PHANTOM[1], 17, style="italic", family=FF_IT, ls=1.5, keep=(keep, occ), pad=6)
    label(Ly.red, px, 2531, PHANTOM[2], 15, weight=500, keep=(keep, occ), pad=6)
    label(Ly.red, px, 2556, PHANTOM[3], 13.5, style="italic", family=FF_IT, keep=(keep, occ), pad=6)


def build_path(Ly, g, keep, occ, rng):
    """朱红小路：一串「·」沿路走，进雾以后渐淡、断掉；「你自己的路」只写一次。"""
    Stream([PATH[0], PATH[1]], rng)                  # 保持主随机序列不变（旧版在这里取过一次）
    r2 = np.random.default_rng(SEED + 300)
    pts = meander(catmull(PATH_CTRL), 5, 300, SEED + 70)
    on = sample(g.land, pts)
    i0 = int(np.argmax(~on)) if (~on).any() else 0     # 港口：第一次离开陆地的地方
    pts = pts[max(0, i0 - 1):]
    pts = pts[pts[:, 0] < FX1 - 30]
    L = Line(smooth_line(pts, 3, False), False)
    Ly.red_defs.append(L.defs(need_rev=False))
    s, step = 9.0, 10.5
    while s < L.L - 2:
        p = L.at(np.array([s]))
        f = float(sample(g.fog, p)[0])
        if f > 0.93:
            break
        keep_p = 1 - smoothstep(0.30, 0.92, f)
        if keep_p >= 1 or r2.random() < keep_p:
            op = 1 - 0.35 * smoothstep(0.2, 0.9, f)
            Ly.red.append(f'<text font-size="24" font-weight="700" text-anchor="middle" fill-opacity="{op:.2f}">'
                          f'<textPath href="#{L.id}" startOffset="{s:.1f}">·</textPath></text>')
        s += step
    keep_poly(keep, L.pts, 8)
    keep_poly(occ, L.pts, 26)
    p0 = L.pts[0]
    Ly.red.append(f'<circle cx="{p0[0]:.1f}" cy="{p0[1]:.1f}" r="5.2"/>')    # 港口
    # 「你自己的路」：写一次，放在路登岸进雾之前那一段的上方（开阔的海面）
    label(Ly.red, 1676, 1262, PATH[0], 27, weight=500, style="italic", ls=4, rot=-2, keep=(keep, occ), pad=8)
    label(Ly.red, 1680, 1296, PATH[1], 16, style="italic", family=FF_IT, ls=1, rot=-2, keep=(keep, occ), pad=6)
    return L


def build_rivers(Ly, g, keep, occ, rng):
    streams = [Stream(items, rng) for items in REGION_ITEMS]
    for k, (ctrl, reg) in enumerate(RIVERS):
        pts = meander(catmull(ctrl), 7, 120, SEED + 80 + k)
        din = sample(g.din, pts)
        inside = din > 12
        if not inside.any():
            continue
        i0 = int(np.argmax(inside))
        i1 = i0 + int(np.argmin(inside[i0:])) if (~inside[i0:]).any() else len(pts)
        pts = pts[i0:i1]
        if len(pts) < 10:
            continue
        L = Line(smooth_line(pts, 2, False), False)
        st = streams[reg]
        drop = make_drop(g, occ, 0.3, 0.9, frame=True)
        Ly.ink += lay_line(L, lambda s: st.next(), lambda u: 5.8 + 2.4 * u, weight=500, gap_em=0.9, mode="free",
                           drop=drop, rng=rng, opacity=0.92, max_bend=0.7)
        Ly.ink_defs.append(L.defs())
        keep_poly(occ, L.pts, 16)


def try_glyph(Ly, g, occ, ch, x, y, sz, weight=500, op=1.0, min_in=10, fog_max=0.9, bw=0.92, bh=0.84):
    hw, hh = 0.5 * sz * bw, 0.5 * sz * bh
    xs, ys = np.array([x - hw, x + hw, x - hw, x + hw, x]), np.array([y - hh, y - hh, y + hh, y + hh, y])
    if not ((xs > FX0 + 20) & (xs < FX1 - 20) & (ys > FY0 + 20) & (ys < FY1 - 20)).all():
        return False
    pts = np.stack([xs, ys], 1)
    if (sample(g.din, pts) < min_in).any():
        return False
    if float(sample(g.fog, pts).mean()) > fog_max:
        return False
    x0, x1, y0, y1 = int(x - hw), int(x + hw) + 1, int(y - hh), int(y + hh) + 1
    if occ[y0:y1, x0:x1].any():
        return False
    occ[y0:y1, x0:x1] = 1
    Ly.ink.append(text_el(x, y, ch, sz, weight=weight, op=op))
    return True


def build_relief(Ly, g, occ, rng):
    """山脉：沿山脊的一条带子里塞满大小不一的「山」，中间高大、两头和两侧矮小；森林：一簇「林」「木」。"""
    n = 0
    for ctrl, smax, hw in RIDGES:
        L = Line(catmull(ctrl), False)

        def frame_at(sv):
            p = L.at(np.array([sv]))[0]
            q = L.at(np.array([min(L.L, sv + 4)]))[0] - L.at(np.array([max(0, sv - 4)]))[0]
            q /= max(np.hypot(*q), 1e-6)
            return p, np.array([-q[1], q[0]])
        # 主峰：沿山脊一个接一个，中间高、两头低，间距不等
        sv = rng.uniform(0, 0.3) * smax
        while sv < L.L:
            u_ = sv / L.L
            env = np.sin(np.pi * u_) ** 0.6
            sz = smax * (0.5 + 0.5 * env) * rng.uniform(0.82, 1.1)
            p, nrm = frame_at(sv)
            pos = p + nrm * rng.normal(0, 0.18) * hw
            w = 600 if sz > 0.8 * smax else 500
            n += try_glyph(Ly, g, occ, "山", pos[0], pos[1], sz, weight=w, min_in=14, bw=1.05, bh=0.92)
            sv += sz * rng.uniform(0.95, 1.5)
        # 山麓：两侧零散的小山
        for _ in range(int(L.L / 9)):
            u_, v = rng.uniform(0, 1), rng.choice([-1, 1]) * rng.uniform(0.45, 1.0)
            env = np.sin(np.pi * u_) ** 0.6
            p, nrm = frame_at(u_ * L.L)
            pos = p + nrm * v * hw * (0.5 + 0.5 * env)
            sz = smax * rng.uniform(0.34, 0.56) * (0.7 + 0.3 * env)
            if sz >= 8.5:
                n += try_glyph(Ly, g, occ, "山", pos[0], pos[1], sz, weight=400, op=0.88, min_in=14, bw=1.25, bh=1.1)
    # 幻岛上那一座小山
    try_glyph(Ly, g, occ, "山", 1012, 2772, 34, weight=700, min_in=4)
    m = 0
    for cx, cy, rx, ry, cnt in FORESTS:
        placed, tries = 0, 0
        while placed < cnt and tries < cnt * 40:
            tries += 1
            a, r = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform(0, 1))
            r *= 1 + 0.25 * np.sin(3 * a + cx)
            x, y = cx + rx * r * np.cos(a), cy + ry * r * np.sin(a)
            ch = "林" if rng.random() < 0.5 else "木"
            sz = rng.uniform(9.5, 13.5) * (1 - 0.25 * r)
            if try_glyph(Ly, g, occ, ch, x, y, sz, weight=500, op=0.95, min_in=18, bw=0.9, bh=0.9):
                placed += 1
        m += placed
    print(f"  山 {n} 个，林木 {m} 个")


def build_coasts(Ly, g, keep, rng):
    streams = [Stream(items, rng) for items in REGION_ITEMS]
    drop = make_drop(g, keep)

    def picker(line):
        def pick(s):
            return streams[int(sample(g.region, line.at(np.array([s])))[0])].next()
        return pick

    drop_lake = make_drop(g, keep, 0.40, 1.02)       # 雾湖：被雾吃掉大半，但岸线还看得出来
    rng_lake = np.random.default_rng(SEED + 310)

    def is_lake(L):
        t = np.gradient(L.pts, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-6)
        return sample(g.lake, L.pts + 6 * np.stack([t[:, 1], -t[:, 0]], 1)).mean() > 0.5

    for L in g.coasts:
        size = 17.0 if L.L > 2400 else (15.5 if L.L > 900 else 13.5)
        if is_lake(L):
            Ly.ink += lay_line(L, picker(L), size, weight=600, drop=drop_lake, rng=rng_lake)
        else:
            Ly.ink += lay_line(L, picker(L), size, weight=600, drop=drop, rng=rng)
        Ly.ink_defs.append(L.defs())
    for L in g.phantom:
        Ly.ink += lay_line(L, picker(L), 15.0, weight=700, drop=None, rng=rng, gap_em=1.15)
        Ly.ink_defs.append(L.defs())
    for k, L in g.rings:
        d, size, op = RINGS[k]
        reg = np.bincount(sample(g.region, L.pts[::7]).astype(int), minlength=8)
        if k == 3 and reg.argmax() == 2:
            continue                                           # 群岛之间少一圈，免得挤
        if reg.argmax() == PH_IDX:
            op = min(1.0, op + 0.12)
        Ly.ink += lay_line(L, picker(L), size, weight=500 if k < 2 else 400, opacity=op, drop=drop, rng=rng)
        Ly.ink_defs.append(L.defs())
    for k, L in g.lake_rings:
        d, size, op = RINGS[k]
        Ly.ink += lay_line(L, picker(L), size, weight=500, opacity=op, drop=drop_lake, rng=rng_lake)
        Ly.ink_defs.append(L.defs())


def build_all(g):
    pass
    rng = np.random.default_rng(SEED)
    Ly = Layers()
    keep = np.zeros((H, W), np.uint8)       # 海岸、水纹要让开的地方
    occ = np.zeros((H, W), np.uint8)        # 陆上山林河要让开的地方
    build_frame(Ly, rng)
    build_cartouche(Ly, keep, rng)
    build_compass(Ly, keep, rng)
    build_scale(Ly, keep, rng)
    build_labels(Ly, g, keep, occ, rng)
    build_path(Ly, g, keep, occ, rng)
    build_rivers(Ly, g, keep, occ, rng)
    build_relief(Ly, g, occ, rng)
    build_coasts(Ly, g, keep, rng)
    return Ly, keep


CSS = "text{dominant-baseline:central}"


def render_layer(body, defs, name):
    grp = f'<g fill="#fff" font-family="{FF}">' + "\n".join(body) + "</g>"
    return svg_mask(grp, defs="\n".join(defs), name=name, css=CSS)


# ================================================================ 纸、做旧、雾（numpy）

PAPER = rgbf("#e8d9b8")


def tint(img, a, col):
    """把 img 往 col（相对于纸色的乘法色）上压 a。"""
    f = col / PAPER
    return img * (1 - a[..., None] * (1 - f))


def stains(rng):
    """水渍：边缘一圈深色潮线，里面略黄；返回 (里面, 潮线) 两层 alpha。"""
    inner = np.zeros((H, W), np.float32)
    ring = np.zeros((H, W), np.float32)
    for k, (cx, cy, r) in enumerate([(1600, 320, 240), (290, 2420, 300), (2140, 2930, 170), (1250, 1180, 110)]):
        x0, x1 = max(0, int(cx - 1.7 * r)), min(W, int(cx + 1.7 * r))
        y0, y1 = max(0, int(cy - 1.7 * r)), min(H, int(cy + 1.7 * r))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        n = fbm(y1 - y0, x1 - x0, SEED + 90 + k, octaves=5, base=r * 0.7)
        n2 = fbm(y1 - y0, x1 - x0, SEED + 95 + k, octaves=4, base=r * 0.25)
        m = 1 - np.hypot(xx - cx, (yy - cy) * 1.1) / r + 0.34 * n
        ins = smoothstep(-0.02, 0.05, m)
        rg = (np.exp(-(m / 0.016) ** 2) * (0.65 + 0.35 * n2)
              + 0.45 * np.exp(-((m - 0.11) / 0.011) ** 2) + 0.25 * np.exp(-((m - 0.26) / 0.009) ** 2))
        inner[y0:y1, x0:x1] = np.maximum(inner[y0:y1, x0:x1], ins * (0.55 + 0.45 * smoothstep(-0.5, 0.5, n2)))
        ring[y0:y1, x0:x1] = np.maximum(ring[y0:y1, x0:x1], rg * smoothstep(-0.1, 0.0, m + 0.1))
    return inner, np.clip(ring, 0, 1)


def foxing(rng):
    """霉斑：几簇大小不一的锈色小点，带一圈晕。"""
    core = np.zeros((H, W), np.float32)
    for cx, cy, n, spread in [(640, 620, 26, 120), (2060, 760, 18, 90), (180, 1900, 22, 70), (1380, 2880, 30, 140),
                              (1780, 210, 14, 80), (980, 1180, 10, 60), (2250, 2050, 16, 70)]:
        for _ in range(n):
            x, y = rng.normal(cx, spread), rng.normal(cy, spread * 0.8)
            r = float(np.clip(rng.lognormal(0.9, 0.55), 1.2, 11))
            a = float(rng.uniform(0.25, 0.75))
            cv2.circle(core, (int(x * 4), int(y * 4)), int(r * 4), a, -1, cv2.LINE_AA, shift=2)
    halo = gaussian(core, 7) * 0.9
    spots = np.clip(gaussian(core, 1.1) + halo, 0, 1)
    return spots


def mold():
    """霉：靠边几片灰绿带棕的斑，里面是细小的颗粒。"""
    out = np.zeros((H, W), np.float32)
    for k, (cx, cy, r) in enumerate([(160, 3010, 210), (2290, 1880, 150), (1500, 3080, 120)]):
        x0, x1 = max(0, int(cx - 1.4 * r)), min(W, int(cx + 1.4 * r))
        y0, y1 = max(0, int(cy - 1.4 * r)), min(H, int(cy + 1.4 * r))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xx - cx, yy - cy) / r
        blob = smoothstep(1.0, 0.3, d + 0.25 * fbm(y1 - y0, x1 - x0, SEED + 121 + k, octaves=3, base=r * 0.5))
        fine = fbm(y1 - y0, x1 - x0, SEED + 125 + k, octaves=3, base=6)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], blob * (0.35 + 0.65 * smoothstep(0.1, 0.9, fine)))
    return out


def creases():
    """两道对折的折痕：竖的是谷折（左暗右亮），横的是山折（上亮下暗）；折线本身有一道暗线和磨损。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nv = fbm(H, W, SEED + 130, octaves=3, base=400)
    xc = W / 2 + 4 * np.sin(yy / 610.0) + 4 * nv
    yc = H / 2 + 4 * np.sin(xx / 540.0 + 1.0) + 4 * nv
    d, e = xx - xc, yy - yc
    shade = (1 + 0.045 * np.tanh(d / 3.0) * np.exp(-(d / 160) ** 2)
             - 0.04 * np.tanh(e / 3.0) * np.exp(-(e / 160) ** 2))
    shade *= 1 - 0.10 * np.exp(-(d / 1.5) ** 2) - 0.09 * np.exp(-(e / 1.5) ** 2)
    shade += 0.035 * np.exp(-((d - 3.2) / 2.0) ** 2) + 0.03 * np.exp(-((e + 3.2) / 2.0) ** 2)
    wn = 0.5 + 0.5 * smoothstep(-0.5, 0.5, fbm(H, W, SEED + 131, octaves=4, base=30))
    wear = np.clip((np.exp(-(d / 4.0) ** 2) + np.exp(-(e / 4.0) ** 2)) * wn, 0, 1)
    hole = np.exp(-(np.hypot(d, e) / 13.0) ** 2)
    wear = np.clip(wear + 0.45 * hole, 0, 1)
    return shade.astype(np.float32), wear.astype(np.float32)


def composite(g, ink, red, fogt):
    rng = np.random.default_rng(SEED + 7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # ---- 纸
    img = paper(H, W, PAPER, seed=SEED + 50, mottle=0.055, fiber=1.1, grain=0.013)
    age = smoothstep(-0.4, 0.9, fbm(H, W, SEED + 51, octaves=5, base=650))
    img = tint(img, 0.38 * age, rgbf("#d3b98a"))
    # ---- 陆地的淡彩：近岸一圈深一点，有水彩的不匀
    landf = gaussian(g.land.astype(np.float32), 1.4)
    wmod = 0.75 + 0.5 * smoothstep(-0.6, 0.6, fbm(H, W, SEED + 52, octaves=5, base=110))
    wash = landf * (0.16 + 0.32 * np.exp(-g.din / 28.0)) * wmod * (1 - 0.85 * g.fog)
    img = tint(img, wash, rgbf("#c09a5e"))
    seab = (1 - landf) * 0.10 * np.exp(-g.dist / 16.0) * wmod * (1 - 0.85 * g.fog)
    img = tint(img, seab, rgbf("#b9a27a"))
    # ---- 折痕（影响墨）
    shade, cwear = creases()
    # ---- 墨
    inkv = 0.88 + 0.12 * smoothstep(-0.6, 0.6, fbm(H, W, SEED + 53, octaves=4, base=340))
    a = np.clip(ink + 0.20 * gaussian(ink, 1.0), 0, 1) * inkv * (1 - 0.6 * cwear)
    inkc = lerp(rgbf("#3f2a1b"), rgbf("#5e4330"), smoothstep(-0.5, 0.7, fbm(H, W, SEED + 54, octaves=4, base=420))[..., None])
    img = img * (1 - a[..., None]) + inkc * a[..., None]
    a = np.clip(red + 0.20 * gaussian(red, 1.0), 0, 1) * (0.92 + 0.08 * inkv) * (1 - 0.6 * cwear)
    img = img * (1 - a[..., None]) + rgbf("#a8402a") * a[..., None]
    # ---- 雾：先把底下糊一点，再盖一层柔白
    wisp = fbm(H, W, SEED + 33, octaves=6, base=220)
    streak = smoothstep(-0.55, 0.75, g.wisp)
    fogv = np.clip(g.fog * (0.74 + 0.40 * streak + 0.08 * wisp), 0, 1)
    img = lerp(img, gaussian(img, 2.4), (0.85 * fogv)[..., None])
    fogc = rgbf("#fbf9f2") * (1 + 0.02 * streak)[..., None]
    img = lerp(img, fogc, (0.88 * fogv)[..., None])
    a = np.clip(gaussian(fogt, 0.5) * 0.78, 0, 1)
    img = img * (1 - a[..., None]) + rgbf("#5a4130") * a[..., None]
    # ---- 水渍、霉斑、霉
    inner, ring = stains(rng)
    img = tint(img, 0.10 * inner, rgbf("#c7a568"))
    img = tint(img, 0.30 * ring, rgbf("#9d7a48"))
    img = tint(img, 0.55 * foxing(rng), rgbf("#9a6436"))
    img = tint(img, 0.28 * mold(), rgbf("#7d7656"))
    # ---- 磨损发暗的边
    e = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy])
    e = np.clip(e + 22 * fbm(H, W, SEED + 56, octaves=5, base=140), 0, None)
    cdist = np.minimum.reduce([np.hypot(xx, yy), np.hypot(W - xx, yy), np.hypot(xx, H - yy), np.hypot(W - xx, H - yy)])
    burn = np.clip(0.50 * np.exp(-e / 75.0) + 0.55 * np.exp(-e / 13.0) + 0.30 * np.exp(-cdist / 300.0), 0, 1)
    img = tint(img, burn, rgbf("#6d4b2a"))
    # ---- 折痕的明暗、磨白，掠射光
    img = img * shade[..., None]
    img = img + 0.025 * cwear[..., None] * (1 - fogv[..., None])
    light = 1.03 - 0.06 * ((xx / W) * 0.6 + (yy / H) * 0.4)
    img = img * light[..., None] * vignette(H, W, strength=0.10, cy=0.48)[..., None]
    return np.clip(img, 0, 1)


def main():
    t0 = time.time()
    g = build_geo()
    for attempt in range(4):
        Ly, keep = build_all(g)
        if not WID.missing:
            break
        print(f"  补量 {len(WID.missing)} 段字宽")
        WID.run()
    print(f"  墨 {len(Ly.ink)} 个元素，朱红 {len(Ly.red)} 个，雾里 {len(Ly.fogt)} 个；排版 {time.time() - t0:.1f}s")
    ink = render_layer(Ly.ink, Ly.ink_defs, "05_ink")
    red = render_layer(Ly.red, Ly.red_defs, "05_red")
    fogt = render_layer(Ly.fogt, [], "05_fogtext")
    img = composite(g, ink, red, fogt)
    save(img, ROOT / "05-图.png", seed=SEED)
    print(f"  用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--land" in sys.argv:
        preview_land()
    else:
        main()
