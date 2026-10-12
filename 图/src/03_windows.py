"""图 3「窗」：夜里的几排楼，每扇亮窗里是一个人此刻在问我的一句话，下面是我回答的开头。

做法：楼、窗、阳台、水箱和天线的形状用 skia 和解析覆盖率画成蒙版；窗里的字交给 Chrome 排——
前排和近排每扇亮窗是一个定宽的 <foreignObject><div>，浏览器负责多语种换行。先用 --dump-dom
量好每句话在每种窗宽、每个字号下排出来有多高，再把句子分给放得下的窗（每句只用一次），挑放得下的
最大字号；中排、远排的窗里是我回答的开头循环写成的小字，只当质感。所有亮窗是同一盏 #ffcf87 的灯：
窗内的灯光渐变、字、窗台的溢光、辉光都只是这一个颜色乘一个亮度。字是房间里最亮的光——窗内底光压得很低，
只对字的蒙版另加一层小半径辉光。雾、天光、混凝土和砖的纹理在 numpy 里合成。

    python -X utf8 03_windows.py
"""
import hashlib
import html as htmlmod
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path

import numpy as np
import skia

sys.path.insert(0, str(Path(__file__).parent))
from common import *          # noqa
from texts_windows import ASKS, REPLIES, REPLY_FOR

SEED = 3
rng = np.random.default_rng(SEED)
LAMP = rgbf("#ffcf87")
OUT = ROOT / "03-窗.png"
T0 = time.time()

# ---------------------------------------------------------------- 句子：每句问话配一句合适的回答开头

_ZH_R = [0, 2, 5, 3, 4, 1, 3, 5, 4, 5, 1, 1, 5, 6, 8, 3, 5, 0, 1, 1, 3, 8, 6, 1, 8, 5, 4, 0, 8, 3, 7, 6, 5,
         4, 8, 4, 0, 8, 1, 5, 7, 1, 5]
_EN_R = [5, 3, 2, 0, 2, 1, 1, 0, 1, 4, 1, 2, 0, 0, 1, 1, 2, 1, 1, 2]


def reply_for_all():
    out, zi, ei = [], 0, 0
    for lang, t in ASKS:
        if lang == "zh":
            out.append(("zh", REPLIES["zh"][_ZH_R[zi]], _ZH_R[zi])); zi += 1
        elif lang == "lat" and t in REPLY_FOR:
            out.append(("lat", REPLY_FOR[t], 100 + len(out)))
        elif lang == "lat":
            out.append(("lat", REPLIES["lat"][_EN_R[ei]], 50 + _EN_R[ei])); ei += 1
        else:
            out.append((lang, REPLIES[lang][0], 200 + len(out)))
    assert zi == len(_ZH_R) and ei == len(_EN_R)
    return out


REPLY = reply_for_all()          # [(lang, text, reply_id)]，与 ASKS 一一对应
HTML_LANG = {"zh": "zh-CN", "ja": "ja", "ko": "ko", "ar": "ar", "hi": "hi", "lat": "en"}

# 远处窗里的质感字：我回答的开头
TEXTURE = [(lang, t) for lang, ts in REPLIES.items() for t in ts] + [("lat", t) for t in REPLY_FOR.values()]

# ---------------------------------------------------------------- 场景数据


@dataclass
class Win:
    gx: float
    gy: float
    gw: float
    gh: float
    layer: int                    # 0 远 1 中 2 近 3 前
    kind: str = "rect"            # rect / arch / transom / door
    frame: float = 5.0
    sill: float = 7.0
    lit: bool = False
    I: float = 1.0
    lamp: tuple = (0.5, 0.3, 100.0)
    curtain: tuple = (0.0, 0.0)   # 左、右窗帘遮住的比例
    text_bottom: float = None     # 文字区下界（阳台门被栏板挡住的部分不放字）
    box: tuple = None             # 文字区 (x, y, w, h)
    ask: int = -1
    fs: float = 0.0
    th: float = 0.0
    off: float = 0.0
    vis: float = 1.0
    floor: int = 0
    col: int = 0
    bid: int = 0
    plant: bool = False
    closed: bool = False


@dataclass
class Bldg:
    x0: float
    x1: float
    top: float
    layer: int
    tone: tuple
    style: str
    wins: list = field(default_factory=list)
    roof: list = field(default_factory=list)      # 楼顶剪影：(fn, args)
    parts: list = field(default_factory=list)     # 立面上的构件：(kind, x0, y0, x1, y1, extra)
    frame_col: tuple = None
    bid: int = 0
    seed: int = 0


# ---------------------------------------------------------------- skia 剪影


def _paint(stroke=0.0):
    p = skia.Paint(AntiAlias=True, Color=skia.ColorWHITE)
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap(skia.Paint.kButt_Cap)
    return p


def r_rect(c, d, x0, y0, x1, y1):
    c.drawRect(skia.Rect.MakeLTRB(x0, y0, x1, y1), _paint())


def r_line(c, d, x0, y0, x1, y1, wdt=2.0):
    c.drawLine(x0, y0, x1, y1, _paint(wdt))


def r_poly(c, d, pts):
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    p.close()
    c.drawPath(p, _paint())


def r_tank(c, d, cx, base, w=118, h=112, leg=72, roof_h=46):
    """纽约式木水箱：四条腿、交叉斜撑、平台、略带收分的桶身、锥顶、箍。"""
    top_leg = base - leg
    for fx in (-0.42, -0.14, 0.14, 0.42):
        r_rect(c, d, cx + fx * w - 3, top_leg, cx + fx * w + 3, base)
    for a, b in ((-0.42, -0.14), (0.14, 0.42), (-0.14, 0.14)):
        r_line(c, d, cx + a * w, base - 4, cx + b * w, top_leg + 6, 2.2)
        r_line(c, d, cx + b * w, base - 4, cx + a * w, top_leg + 6, 2.2)
    r_line(c, d, cx - 0.45 * w, base - leg * 0.5, cx + 0.45 * w, base - leg * 0.5, 2.5)
    r_rect(c, d, cx - 0.56 * w, top_leg - 9, cx + 0.56 * w, top_leg)
    yb, yt = top_leg - 9, top_leg - 9 - h
    r_poly(c, d, [(cx - w / 2, yb), (cx + w / 2, yb), (cx + w * 0.46, yt), (cx - w * 0.46, yt)])
    r_poly(c, d, [(cx - w * 0.52, yt + 2), (cx + w * 0.52, yt + 2), (cx + 4, yt - roof_h), (cx - 4, yt - roof_h)])
    r_rect(c, d, cx - 2.5, yt - roof_h - 14, cx + 2.5, yt - roof_h + 2)
    c.drawCircle(cx, yt - roof_h - 15, 4.5, _paint())
    # 箍和梯子（细节层：比剪影略亮）
    for k in range(5):
        yy = yb - (k + 0.6) * h / 5.6
        ww = w / 2 - (yb - yy) / h * w * 0.04
        d.drawLine(cx - ww, yy, cx + ww, yy, _paint(1.6))
    d.drawLine(cx + w * 0.30, yb + 2, cx + w * 0.27, yt + 4, _paint(1.2))
    d.drawLine(cx + w * 0.38, yb + 2, cx + w * 0.35, yt + 4, _paint(1.2))
    for k in range(9):
        yy = yb - k * h / 9
        d.drawLine(cx + w * 0.30 - k * 0.3, yy, cx + w * 0.38 - k * 0.3, yy, _paint(1.0))


def r_antenna(c, d, x, base, h, bars=((0.75, 60), (0.55, 44), (0.35, 30)), wdt=3.0, stay=True):
    r_line(c, d, x, base, x, base - h, wdt)
    for fy, L in bars:
        y = base - h * fy
        r_line(c, d, x - L / 2, y, x + L / 2, y, 2.0)
        for k in range(5):
            xx = x - L / 2 + (k + 0.5) * L / 5
            r_line(c, d, xx, y - 5, xx, y + 5, 1.2)
    if stay:
        r_line(c, d, x, base - h * 0.6, x - h * 0.45, base, 0.9)
        r_line(c, d, x, base - h * 0.6, x + h * 0.38, base, 0.9)


def r_mast(c, d, x, base, h):
    """细长的天线杆：上细下粗，两层横担，顶上一根针。"""
    r_poly(c, d, [(x - 4, base), (x + 4, base), (x + 1.5, base - h), (x - 1.5, base - h)])
    for fy, L in ((0.55, 70), (0.78, 46)):
        y = base - h * fy
        r_line(c, d, x - L / 2, y, x + L / 2, y, 2.4)
        for s in (-1, 1):
            r_line(c, d, x + s * L / 2, y, x + s * L / 2, y - 22, 1.6)
    r_line(c, d, x, base - h, x, base - h - 40, 1.2)
    r_line(c, d, x, base - h * 0.5, x - h * 0.30, base, 0.9)
    r_line(c, d, x, base - h * 0.5, x + h * 0.33, base, 0.9)


def r_dish(c, d, cx, base, r=26):
    r_rect(c, d, cx - 2, base - r * 0.9, cx + 2, base)
    c.save()
    c.translate(cx, base - r * 1.25)
    c.rotate(-28)
    c.drawOval(skia.Rect.MakeLTRB(-r * 0.42, -r, r * 0.42, r), _paint())
    c.restore()
    r_line(c, d, cx - r * 0.1, base - r * 1.25, cx + r * 0.75, base - r * 1.7, 1.6)


def r_chimney(c, d, x0, x1, base, h):
    r_rect(c, d, x0, base - h, x1, base)
    r_rect(c, d, x0 - 5, base - h - 7, x1 + 5, base - h)
    r_rect(c, d, x0 + 6, base - h - 17, x0 + 14, base - h - 7)


def r_railing(c, d, x0, x1, base, h=34, step=13):
    r_line(c, d, x0, base - h, x1, base - h, 3.0)
    r_line(c, d, x0, base - h * 0.45, x1, base - h * 0.45, 1.4)
    x = x0
    while x <= x1:
        r_line(c, d, x, base - h, x, base, 2.0)
        x += step


def r_box(c, d, x0, x1, base, h, lip=6):
    r_rect(c, d, x0, base - h, x1, base)
    r_rect(c, d, x0 - lip, base - h - 6, x1 + lip, base - h)


ROOF_FN = {"rect": r_rect, "line": r_line, "poly": r_poly, "tank": r_tank, "antenna": r_antenna,
           "mast": r_mast, "dish": r_dish, "chimney": r_chimney, "railing": r_railing, "box": r_box}


def draw_sil(b, ox, oy, w, h, with_body=True):
    """返回 (整体剪影, 楼顶附属物, 附属物的细节线) 三个 [0,1] 蒙版。"""
    surfs = [skia.Surface(w, h) for _ in range(3)]
    cvs = [s.getCanvas() for s in surfs]
    junk = skia.Surface(4, 4).getCanvas()
    for cv in cvs:
        cv.clear(skia.ColorBLACK)
        cv.translate(-ox, -oy)
    if with_body:
        cvs[0].drawRect(skia.Rect.MakeLTRB(b.x0, b.top, b.x1, H + 10), _paint())
    for name, args in b.roof:
        ROOF_FN[name](cvs[0], cvs[2], *args)
        ROOF_FN[name](cvs[1], junk, *args)
    out = [s.makeImageSnapshot().toarray()[..., 0].astype(np.float32) / 255.0 for s in surfs]
    return out


# ---------------------------------------------------------------- 建筑


def lit_pattern(n_floor, n_col, frac, seed, row_w=0.7, col_w=0.45, bias=None):
    """有成团倾向的亮灯分布：每层、每列各有一个倾向，加上每扇窗自己的随机；bias 是每层额外的倾向。"""
    r = np.random.default_rng(seed)
    v = r.normal(0, 1, (n_floor, n_col)) + row_w * r.normal(0, 1, (n_floor, 1)) + col_w * r.normal(0, 1, (1, n_col))
    if bias is not None:
        v = v + np.asarray(bias, np.float32)[:n_floor, None]
    thr = np.quantile(v, 1 - frac) if v.size else 0
    return v > thr


def lamp_for(win, r):
    lx = win.gx + win.gw * r.uniform(0.12, 0.88)
    ly = win.gy + win.gh * r.uniform(-0.15, 0.75)
    rad = max(win.gw, win.gh) * r.uniform(0.38, 0.80)
    return (lx, ly, rad)


def front_buildings():
    bs = []

    # F1 老砖楼：孟莎式屋顶上两扇老虎窗，下面是拱顶窄高窗，檐口、烟囱
    T1 = 1606
    b = Bldg(-40, 452, T1, 3, (0.056, 0.042, 0.044), "brick", seed=101)
    b.roof = [("poly", ([(-46, T1 - 4), (-6, T1 - 150), (412, T1 - 150), (458, T1 - 4)],)),
              ("rect", (-52, T1 - 18, 464, T1 + 4)), ("rect", (-10, T1 - 160, 416, T1 - 148)),
              ("chimney", (60, 98, T1 - 150, 70)), ("chimney", (338, 370, T1 - 150, 56)),
              ("antenna", (79, T1 - 220, 130, ((0.8, 70), (0.6, 52), (0.42, 38)), 3.0, False)),
              ("line", (79, T1 - 232, 20, T1 - 150, 0.9))]
    for cx in (112, 318):          # 老虎窗：带三角山花的小凸窗
        b.roof += [("rect", (cx - 58, T1 - 128, cx + 58, T1 - 4)),
                   ("poly", ([(cx - 68, T1 - 124), (cx + 68, T1 - 124), (cx, T1 - 176)],))]
        b.wins.append(Win(cx - 44, T1 - 112, 88, 96, 3, frame=5, sill=6, floor=0, col=(0 if cx < 200 else 1)))
    gy0, fh = T1 + 70, 258
    for k in range(7):
        gy = gy0 + k * fh
        if gy + 200 > H - 24:
            break
        for j, cx in enumerate((112, 318)):
            b.wins.append(Win(cx - 51, gy, 102, 204, 3, kind="arch", frame=6, sill=9, floor=k + 1, col=j))
        b.parts.append(("course", -40, gy + 204 + 15, 452, gy + 204 + 24, None))
    bs.append(b)

    # F3 矮的抹灰楼：方窗、空调外机、落水管，楼顶有天线和锅
    T3 = 1884
    b = Bldg(1028, 1432, T3, 3, (0.058, 0.057, 0.063), "stucco", seed=303)
    b.roof = [("rect", (1028, T3 - 22, 1432, T3)),
              ("railing", (1040, 1420, T3 - 22, 34, 12)),
              ("antenna", (1110, T3 - 22, 128)),
              ("antenna", (1176, T3 - 22, 96, ((0.8, 48), (0.55, 34)), 2.4)),
              ("dish", (1340, T3 - 22, 24)),
              ("box", (1250, 1300, T3 - 22, 38, 4))]
    gy0, fh = T3 + 68, 222
    for k in range(8):
        gy = gy0 + k * fh
        if gy + 134 > H - 24:
            break
        for j, cx in enumerate((1132, 1328)):
            b.wins.append(Win(cx - 64, gy, 128, 134, 3, frame=5, sill=8, floor=k, col=j))
    b.parts.append(("pipe", 1230, T3, 1238, H + 4, None))
    bs.append(b)

    # F4 阳台楼：每层两开间，阳台门 + 小窗，实心栏板
    T4 = 1296
    b = Bldg(1404, 2022, T4, 3, (0.045, 0.049, 0.059), "panel", seed=404)
    b.roof = [("rect", (1404, T4 - 22, 2022, T4)),
              ("railing", (1412, 1700, T4 - 22, 30, 14)),
              ("box", (1520, 1640, T4 - 22, 64, 6)),
              ("mast", (1912, T4 - 22, 270)),
              ("antenna", (1770, T4 - 22, 84, ((0.8, 44), (0.55, 30)), 2.2))]
    fh = 268
    rp = np.random.default_rng(4041)
    for k in range(8):
        yf = T4 + 300 + k * fh
        if yf - 226 > H - 60:
            break
        for j, (bx0, bx1, mirror) in enumerate(((1410, 1712, False), (1714, 2016, True))):
            bcx = (bx0 + bx1) / 2
            dcx, wcx = (bcx - 70, bcx + 74) if not mirror else (bcx + 70, bcx - 74)
            dwin = Win(dcx - 59, yf - 228, 118, 216, 3, kind="door", frame=5, sill=0, floor=k, col=2 * j)
            dwin.text_bottom = yf - 92
            wwin = Win(wcx - 46, yf - 222, 92, 120, 3, frame=5, sill=0, floor=k, col=2 * j + 1)
            b.wins += [dwin, wwin]
            b.parts.append(("slab", bx0, yf, bx1, yf + 16, None))
            plants = []
            if rp.random() < 0.42:      # 栏板上摆着几盆花草，挡在亮着的门前
                n = int(rp.integers(1, 4))
                for q in range(n):
                    plants.append((dcx + rp.uniform(-48, 48), rp.uniform(0.8, 1.25), str(rp.choice(["bush", "spiky", "trail"])),
                                   int(rp.integers(0, 1 << 30))))
                dwin.text_bottom = yf - 92 - 50
            b.parts.append(("parapet", bx0 + 6, yf - 92, bx1 - 6, yf, (dwin, wwin, plants)))
    bs.append(b)

    # F2 高的混凝土楼：带上亮子的宽窗，楼板线，楼顶水箱和楼梯间
    b = Bldg(418, 1052, 1052, 3, (0.048, 0.052, 0.062), "concrete", seed=202)
    b.roof = [("rect", (412, 1028, 1058, 1052)),
              ("box", (470, 600, 1028, 104, 7)),
              ("tank", (878, 1028, 124, 116, 76, 48)),
              ("antenna", (640, 1028, 110, ((0.8, 54), (0.58, 40), (0.38, 28)), 2.6))]
    gy0, fh = 1124, 236
    for k in range(9):
        gy = gy0 + k * fh
        if gy + 168 > H - 24:
            break
        for j, cx in enumerate((582, 888)):
            b.wins.append(Win(cx - 111, gy, 222, 168, 3, kind="transom", frame=6, sill=8, floor=k, col=j))
        b.parts.append(("band", 418, gy - 46, 1052, gy - 32, None))
    bs.append(b)

    # F5 带壁柱的楼：山花女儿墙，每层三扇窄窗，楼顶大水箱和天线
    T5 = 1540
    b = Bldg(1994, 2444, T5, 3, (0.050, 0.050, 0.060), "pilaster", seed=505)
    b.roof = [("rect", (1988, T5 - 22, 2450, T5)),
              ("poly", ([(2120, T5 - 20), (2318, T5 - 20), (2219, T5 - 74)],)),
              ("tank", (2370, T5 - 22, 112, 104, 60, 44)),
              ("antenna", (2060, T5 - 22, 140, ((0.82, 66), (0.62, 50), (0.44, 36)), 3.0))]
    gy0, fh = T5 + 74, 232
    for k in range(8):
        gy = gy0 + k * fh
        if gy + 172 > H - 24:
            break
        for j, cx in enumerate((2082, 2218, 2354)):
            b.wins.append(Win(cx - 46, gy, 92, 172, 3, frame=5, sill=8, floor=k, col=j))
    for px in (2150, 2286):
        b.parts.append(("pilaster", px - 11, T5, px + 11, H + 4, None))
    bs.append(b)

    # 亮灯
    for bi, b in enumerate(bs):
        b.bid = 30 + bi
        nf = max(w.floor for w in b.wins) + 1
        nc = max(w.col for w in b.wins) + 1
        if b.style == "concrete":     # 最高那栋的上面几层多亮几盏，压住左上那一大块暗
            pat = lit_pattern(nf, nc, 0.68, b.seed, bias=[1.2, 1.6, 0.9] + [0.0] * nf)
        else:
            pat = lit_pattern(nf, nc, 0.60, b.seed)
        r = np.random.default_rng(b.seed + 7)
        for w in b.wins:
            w.bid = b.bid
            w.lit = bool(pat[w.floor, w.col])
            w.I = float(np.clip(r.lognormal(0.06, 0.34), 0.42, 1.30)) * (1.10 if r.random() < 0.08 else 1.0)
            w.lamp = lamp_for(w, r)
            w.closed = (not w.lit) and r.random() < 0.28
            if w.kind in ("rect", "arch") and r.random() < 0.30:
                side = r.integers(0, 3)
                f = r.uniform(0.16, 0.28)
                w.curtain = (f, 0.0) if side == 0 else (0.0, f) if side == 1 else (f * 0.6, f * 0.6)
        b.frame_col = None
    return bs


def proc_building(x0, x1, top, layer, seed, tone, gw, gh, sx, sy, frac, frame, mtop=None, roof=()):
    r = np.random.default_rng(seed)
    b = Bldg(x0, x1, top, layer, tone, "plain", seed=seed)
    b.roof = list(roof)
    ncol = max(1, int((x1 - x0 - 2 * frame - 6) // sx))
    used = ncol * sx - (sx - gw)
    mx = (x1 - x0 - used) / 2
    y = top + (mtop if mtop is not None else sy * 0.6)
    k = 0
    rows = []
    while y + gh < H:
        rows.append(y)
        y += sy
    pat = lit_pattern(len(rows), ncol, frac, seed + 1)
    for k, yy in enumerate(rows):
        for j in range(ncol):
            w = Win(x0 + mx + j * sx, yy, gw, gh, layer, frame=frame, sill=frame * 1.4, floor=k, col=j)
            w.lit = bool(pat[k, j])
            w.I = float(np.clip(r.lognormal(0, 0.25), 0.5, 1.6))
            w.lamp = lamp_for(w, r)
            b.wins.append(w)
    return b


def near_buildings():
    out = []
    specs = [  # x0, x1, top, gw, gh, sx, sy, frac, tone
        (30, 330, 1300, 52, 62, 72, 88, 0.62, (0.060, 0.066, 0.084)),
        (1066, 1394, 1150, 58, 66, 78, 94, 0.58, (0.057, 0.064, 0.082)),
        (1690, 1990, 1040, 50, 60, 68, 86, 0.55, (0.059, 0.066, 0.085)),
        (2140, 2420, 1250, 54, 62, 74, 90, 0.58, (0.061, 0.067, 0.085)),
    ]
    roofs = [
        [("rect", (24, 1286, 336, 1300)), ("antenna", (110, 1286, 70, ((0.8, 30), (0.55, 22)), 1.6))],
        [("rect", (1060, 1136, 1400, 1150)), ("tank", (1310, 1136, 64, 60, 38, 26)),
         ("railing", (1072, 1240, 1136, 18, 8))],
        [("rect", (1684, 1026, 1996, 1040)), ("mast", (1790, 1026, 120))],
        [("rect", (2134, 1236, 2426, 1250)), ("box", (2300, 2360, 1236, 30, 4))],
    ]
    for i, (x0, x1, top, gw, gh, sx, sy, frac, tone) in enumerate(specs):
        b = proc_building(x0, x1, top, 2, 700 + i * 11, tone, gw, gh, sx, sy, frac, 3.0, mtop=34, roof=roofs[i])
        b.bid = 20 + i
        for w in b.wins:
            w.bid = b.bid
            w.sill = 4.0
            w.I = min(w.I, 1.3)
        out.append(b)
    return out


def mid_buildings():
    specs = [  # x0, x1, top
        (-30, 214, 1046), (170, 396, 1170), (330, 560, 858), (548, 744, 992), (712, 952, 936),
        (900, 1102, 1004), (1044, 1250, 886), (1236, 1470, 972), (1436, 1690, 1036),
        (1940, 2190, 938), (2150, 2330, 1084), (2290, 2450, 968),
    ]
    out = []
    r = np.random.default_rng(55)
    for i, (x0, x1, top) in enumerate(specs):
        gw = r.uniform(24, 34)
        gh = gw * r.uniform(1.15, 1.5)
        sx = gw + r.uniform(9, 16)
        sy = gh + r.uniform(10, 18)
        tone = tuple(np.array((0.062, 0.070, 0.094)) * r.uniform(0.85, 1.12))
        roof = [("rect", (x0 - 3, top - 8, x1 + 3, top))]
        if r.random() < 0.5:
            ax = r.uniform(x0 + 20, x1 - 20)
            roof.append(("antenna", (ax, top - 8, r.uniform(40, 90), ((0.8, 26), (0.55, 18)), 1.6, False)))
        if r.random() < 0.35:
            bx = r.uniform(x0 + 10, x1 - 70)
            roof.append(("box", (bx, bx + r.uniform(30, 60), top - 8, r.uniform(18, 34), 3)))
        if i == 2:   # 中排最高的一栋：顶上收分 + 长天线
            roof += [("rect", (x0 + 30, top - 60, x1 - 30, top - 6)), ("rect", (x0 + 70, top - 96, x1 - 70, top - 58)),
                     ("mast", ((x0 + x1) / 2, top - 96, 150))]
        b = proc_building(x0, x1, top, 1, 900 + i * 13, tone, gw, gh, sx, sy, r.uniform(0.45, 0.6), 1.6, roof=roof)
        b.bid = 10 + i
        for w in b.wins:
            w.bid = b.bid
        out.append(b)
    return out


def far_buildings():
    out = []
    r = np.random.default_rng(77)
    x = -50.0
    i = 0
    while x < W + 40:
        w = r.uniform(70, 180)
        top = 1040 + 150 * np.sin(x / 310 + 1.3) + r.normal(0, 70)
        if r.random() < 0.18:
            top -= r.uniform(120, 220)
        top = float(np.clip(top, 820, 1330))
        gw = r.uniform(7, 11)
        gh = gw * r.uniform(1.1, 1.6)
        sx = gw + r.uniform(4, 8)
        sy = gh + r.uniform(5, 9)
        tone = tuple(np.array((0.075, 0.083, 0.112)) * r.uniform(0.85, 1.12))
        roof = [("rect", (x - 2, top - 5, x + w + 2, top))]
        if r.random() < 0.35:
            roof.append(("line", (x + w * r.uniform(0.2, 0.8), top - 5, x + w * r.uniform(0.2, 0.8) + 0.01,
                                  top - r.uniform(25, 70), 1.2)))
        b = proc_building(x, x + w, top, 0, 2000 + i, tone, gw, gh, sx, sy, r.uniform(0.3, 0.5), 0.8, roof=roof)
        b.bid = 100 + i
        for ww in b.wins:
            ww.bid = b.bid
        out.append(b)
        x += w * r.uniform(0.55, 1.0)
        i += 1
    # 远处一栋细高的塔楼，顶上收分加一根针
    rt = np.random.default_rng(78)
    x0t, x1t, topt = 1604.0, 1724.0, 600.0
    roof = [("rect", (x0t - 3, topt - 6, x1t + 3, topt)), ("rect", (x0t + 14, topt - 46, x1t - 14, topt - 5)),
            ("rect", (x0t + 34, topt - 80, x1t - 34, topt - 45)), ("line", (1664, topt - 80, 1664.01, topt - 200, 2.0))]
    bt = proc_building(x0t, x1t, topt, 0, 2999, (0.072, 0.080, 0.108), 9.5, 13.0, 15.0, 20.0, 0.42, 0.8, roof=roof)
    bt.bid = 199
    for ww in bt.wins:
        ww.bid = bt.bid
    out.append(bt)
    order = r.permutation(len(out))      # 远排不按从左到右叠
    return [out[k] for k in order]


# ---------------------------------------------------------------- 可见性


def union_mask(bs):
    s = skia.Surface(W, H)
    c = s.getCanvas()
    c.clear(skia.ColorBLACK)
    for b in bs:
        c.drawRect(skia.Rect.MakeLTRB(b.x0, b.top, b.x1, H + 10), _paint())
        for name, args in b.roof:
            ROOF_FN[name](c, c, *args)
    return s.makeImageSnapshot().toarray()[..., 0].astype(np.float32) / 255.0


def set_visibility(wins, occ):
    for w in wins:
        x0, y0 = int(w.gx), int(w.gy)
        x1, y1 = int(ceil(w.gx + w.gw)), int(ceil(w.gy + w.gh))
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            w.vis = 0.0
            continue
        w.vis = float(1 - occ[y0:y1, x0:x1].mean())


# ---------------------------------------------------------------- 排字：量、分、排

CSS = """
.win{color:#fff;line-height:1.36;overflow-wrap:normal;word-break:normal;line-break:strict;
 text-rendering:geometricPrecision;font-kerning:normal;margin:0;padding:0;white-space:normal}
.win .a{font-weight:500}
.win .r{color:#9a9a9a;font-size:0.8em;line-height:1.34;margin-top:0.5em;font-weight:400}
.win.ko{word-break:keep-all}
"""
FRONT_FS = [14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0]
NEAR_FS = [7.5, 7.0, 6.5, 6.0, 5.5]


def win_div(i, width, fs, extra=""):
    lang, ask = ASKS[i]
    rlang, rep, _ = REPLY[i]
    d = ' dir="rtl"' if lang == "ar" else ' dir="ltr"'
    cls = "win ko" if lang == "ko" else "win"
    return (f'<div xmlns="http://www.w3.org/1999/xhtml" class="{cls}" lang="{HTML_LANG[lang]}"{d} {extra}'
            f'style="width:{width:.1f}px;font-size:{fs:.2f}px">'
            f'<div class="a" style="font-family:{esc(FONT[lang])}">{esc(ask)}</div>'
            f'<div class="r" style="font-family:{esc(FONT[rlang])}">{esc(rep)}</div></div>')


def chrome_dom(html, name):
    path = WORK / f"{name}.html"
    path.write_text(html, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="chrome-ud-") as ud:
        t = time.time()
        r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={ud}",
                            "--virtual-time-budget=20000", "--window-size=1600,1200", "--dump-dom",
                            path.as_uri()], capture_output=True, timeout=600)
    out = r.stdout.decode("utf-8", "replace")
    print(f"  chrome dom {name}: {time.time() - t:.1f}s")
    return out


def measure(jobs):
    """jobs: [(ask_idx, width, fs)] → {key: (height, overflow)}；带缓存（CSS 或字体栈变了就作废）。"""
    tag = hashlib.md5((CSS + json.dumps(FONT, sort_keys=True)).encode()).hexdigest()
    cache_p = WORK / "03_measure_cache.json"
    cache = {}
    if cache_p.exists():
        d = json.loads(cache_p.read_text(encoding="utf-8"))
        if d.get("tag") == tag:
            cache = d["data"]
    key = lambda i, w, f: f"{i}|{w:.1f}|{f:.2f}"
    todo = sorted({key(*j) for j in jobs} - set(cache))
    if todo:
        divs = []
        for k in todo:
            i, w, f = k.split("|")
            divs.append(win_div(int(i), float(w), float(f), extra=f'data-k="{k}" '))
        page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;padding:0;background:#000}}
{CSS}
</style></head><body>
{chr(10).join(divs)}
<script>
document.fonts.ready.then(()=>{{const out={{}};
for(const e of document.querySelectorAll('.win[data-k]')){{const r=e.getBoundingClientRect();
out[e.dataset.k]=[Math.round(r.height*100)/100, e.scrollWidth>e.clientWidth+0.5?1:0];}}
const p=document.createElement('pre');p.id='result';p.textContent=JSON.stringify(out);document.body.appendChild(p);}});
</script></body></html>"""
        dom = chrome_dom(page, "03_measure")
        m = re.search(r'<pre id="result">(.*?)</pre>', dom, re.S)
        if not m:
            raise RuntimeError("量字失败：dump-dom 里没有结果")
        got = json.loads(htmlmod.unescape(m.group(1)))
        cache.update(got)
        cache_p.write_text(json.dumps({"tag": tag, "data": cache}), encoding="utf-8")
        print(f"  量了 {len(todo)} 个排版")
    return {k: cache[k] for k in {key(*j) for j in jobs}}, key


def text_box(w):
    """窗里放字的区域：去掉窗帘、上亮子、栏板挡住的部分，留一点边。"""
    if w.layer == 3:
        px, py = max(8.0, w.gw * 0.075), 10.0
    else:
        px, py = 4.0, 4.5
    x0 = w.gx + w.gw * w.curtain[0] + px
    x1 = w.gx + w.gw * (1 - w.curtain[1]) - px
    y0 = w.gy + py
    if w.kind == "arch":
        y0 = w.gy + w.gw * 0.5 + 2
    if w.kind == "transom":
        y0 = w.gy + w.gh * 0.24 + 6 + py
    y1 = (w.text_bottom if w.text_bottom else w.gy + w.gh) - py
    return (x0, y0, round((x1 - x0) * 2) / 2, y1 - y0)


def assign_texts(front_w, near_w):
    """按量好的高度把每句问话分给一扇放得下的窗（每句只用一次），字号取放得下的最大。"""
    for w in front_w + near_w:
        w.box = text_box(w)
    jobs = []
    for w in front_w:
        for i in range(len(ASKS)):
            for f in FRONT_FS:
                jobs.append((i, w.box[2], f))
    for w in near_w:
        for i in range(len(ASKS)):
            for f in NEAR_FS:
                jobs.append((i, w.box[2], f))
    res, key = measure(jobs)

    def best_fs(i, w):
        sizes = FRONT_FS if w.layer == 3 else NEAR_FS
        for f in sizes:
            hgt, ovf = res[key(i, w.box[2], f)]
            if not ovf and hgt <= w.box[3] * 0.97:
                return f, hgt
        return None, None

    r = np.random.default_rng(SEED + 99)
    wins = front_w + near_w
    feas = {id(w): [i for i in range(len(ASKS)) if best_fs(i, w)[0]] for w in wins}
    order = sorted(wins, key=lambda w: (len(feas[id(w)]) + r.uniform(0, 6), -w.layer))
    used = set()
    placed = []
    for w in order:
        cands = [i for i in feas[id(w)] if i not in used]
        if not cands:
            w.lit = False
            continue
        nb = [p for p in placed if p.bid == w.bid and abs(p.floor - w.floor) <= 1 and abs(p.col - w.col) <= 1]
        nb_lang = {ASKS[p.ask][0] for p in nb}
        nb_rep = {REPLY[p.ask][2] for p in nb}
        best, bi = -1e9, None
        for i in cands:
            f, hgt = best_fs(i, w)
            sizes = FRONT_FS if w.layer == 3 else NEAR_FS
            s = 1.2 * (f - sizes[-1]) / (sizes[0] - sizes[-1])     # 字大一点好
            s += 1.0 * min(1.0, hgt / w.box[3] / 0.55)               # 窗里别太空
            s -= 0.9 * (ASKS[i][0] in nb_lang and ASKS[i][0] != "zh")
            s -= 0.7 * (REPLY[i][2] in nb_rep)
            s += r.normal(0, 0.45)
            if s > best:
                best, bi = s, i
        w.ask = bi
        w.fs, w.th = best_fs(bi, w)
        slack = w.box[3] - w.th
        w.off = float(slack * r.uniform(0.0, 0.42) if w.layer == 3 else slack * r.uniform(0.0, 0.3))
        used.add(bi)
        placed.append(w)
    print(f"  亮窗：前排 {sum(1 for w in front_w if w.lit)}，近排 {sum(1 for w in near_w if w.lit)}，"
          f"用掉 {len(used)}/{len(ASKS)} 句")
    return used


def build_window_text(wins):
    out = []
    for w in wins:
        if not w.lit or w.ask < 0:
            continue
        x, y, bw, bh = w.box
        out.append(f'<foreignObject x="{x:.2f}" y="{y + w.off:.2f}" width="{bw + 2:.1f}" height="{bh + 4:.1f}">'
                   f'{win_div(w.ask, bw, w.fs)}</foreignObject>')
    return "\n".join(out)


def build_texture_text(wins, size, lead, seed):
    r = np.random.default_rng(seed)
    out = []
    for w in wins:
        if not w.lit or w.vis < 0.02:
            continue
        s = size * r.uniform(0.92, 1.08)
        y = w.gy + s * 1.05 + r.uniform(0, 0.6)
        while y < w.gy + w.gh + s:
            lang, t = TEXTURE[int(r.integers(0, len(TEXTURE)))]
            lang2, t2 = TEXTURE[int(r.integers(0, len(TEXTURE)))]
            out.append(f'<text x="{w.gx + 0.4 - r.uniform(0, 3 * s):.2f}" y="{y:.2f}" font-size="{s:.2f}" fill="#fff" '
                       f'xml:space="preserve">{tspans([(lang, t), (lang2, t2)], sep=" ")}</text>')
            y += s * lead
    return "\n".join(out)


def build_stars():
    pts = [(310, 210, 7.0, 0.8), (1530, 120, 6.0, 0.7), (2010, 380, 6.5, 0.75), (860, 520, 5.5, 0.55),
           (2260, 150, 5.5, 0.6)]
    picks = [("zh", REPLIES["zh"][0]), ("lat", REPLIES["lat"][0]), ("ja", REPLIES["ja"][0]),
             ("zh", REPLIES["zh"][4]), ("ko", REPLIES["ko"][0])]
    out = []
    for (x, y, s, op), (lang, t) in zip(pts, picks):
        out.append(f'<text x="{x}" y="{y}" font-size="{s}" font-family="{FONT[lang]}" fill="#fff" '
                   f'fill-opacity="{op}" text-anchor="middle">{esc(t)}</text>')
    return "\n".join(out), pts


# ---------------------------------------------------------------- 合成


def shift(a, dy, dx):
    """不回卷的平移（空出来的地方补边缘值）。"""
    out = np.empty_like(a)
    h, w = a.shape[:2]
    src = a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    if dy > 0:
        out[:dy] = out[dy:dy + 1]
    if dx > 0:
        out[:, :dx] = out[:, dx:dx + 1]
    return out


def cov_rect(x0, y0, x1, y1, ox, oy, w, h):
    xs = np.arange(w, dtype=np.float32) + ox
    ys = np.arange(h, dtype=np.float32) + oy
    cx = np.clip(np.minimum(xs + 1, x1) - np.maximum(xs, x0), 0, 1)
    cy = np.clip(np.minimum(ys + 1, y1) - np.maximum(ys, y0), 0, 1)
    return cy[:, None] * cx[None, :]


def cov_arch(gx, gy, gw, gh, ox, oy, w, h, grow=0.0):
    X = np.arange(w, dtype=np.float32)[None, :] + ox + 0.5
    Y = np.arange(h, dtype=np.float32)[:, None] + oy + 0.5
    r = gw / 2 + grow
    cx, cy = gx + gw / 2, gy + gw / 2
    dbox = np.maximum(np.abs(X - cx) - r, np.maximum(cy - Y, Y - (gy + gh + grow)))
    dcir = np.maximum(np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r, Y - cy)
    return np.clip(0.5 - np.minimum(dbox, dcir), 0, 1)


def hash2(a, b):
    v = np.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - np.floor(v)


def facade(b, ox, oy, w, h):
    X = np.arange(w, dtype=np.float32)[None, :] + ox + 0.5
    Y = np.arange(h, dtype=np.float32)[:, None] + oy + 0.5
    big = fbm(h, w, b.seed, octaves=5, base=220 if b.layer == 3 else 120)
    fine = fbm(h, w, b.seed + 1, octaves=3, base=7)
    amp = 1.6 if b.layer == 3 else 1.0
    lum = 1 + amp * (0.08 * big + 0.04 * fine)
    if b.style == "brick":
        ch, bw_ = 9.0, 25.0
        row = np.floor(Y / ch)
        xo = X + (row % 2) * bw_ / 2
        col = np.floor(xo / bw_)
        my = np.clip(1.3 - np.abs((Y % ch) - 0.5) * 1.6, 0, 1)
        mx = np.clip(1.3 - np.abs((xo % bw_) - 0.5) * 1.6, 0, 1)
        mortar = np.maximum(my, mx)
        lum = lum * (1 - 0.10 * mortar + 0.07 * (hash2(row, col) - 0.5))
    elif b.style == "concrete":
        jx = np.clip(1.2 - np.abs(((X - b.x0) % 158) - 1) * 1.2, 0, 1)
        pr = np.floor((X - b.x0) / 158)
        lum = lum * (1 - 0.05 * jx + 0.035 * (hash2(pr, 3.0) - 0.5))
    elif b.style == "stucco":
        lum = lum + 0.05 * fbm(h, w, b.seed + 5, octaves=4, base=40)
    elif b.style == "panel":
        jy = np.clip(1.2 - np.abs(((Y - b.top) % 134) - 1) * 1.2, 0, 1)
        lum = lum * (1 - 0.04 * jy)
    # 雨水在窗台下冲出的竖条印子
    st = np.zeros((h, w), np.float32)
    rr = np.random.default_rng(b.seed + 11)
    noise_cols = fbm(h, w, b.seed + 12, octaves=4, base=30)
    for win in b.wins:
        if win.layer < 2:
            continue
        ys = win.gy + win.gh + win.frame + win.sill
        L = rr.uniform(60, 220)
        cx = win.gx + win.gw * rr.uniform(0.3, 0.7)
        sw = win.gw * rr.uniform(0.18, 0.4)
        x0i, x1i = int(max(0, cx - 3 * sw - ox)), int(min(w, cx + 3 * sw - ox))
        y0i, y1i = int(max(0, ys - oy)), int(min(h, ys + 3 * L - oy))
        if x1i <= x0i or y1i <= y0i:
            continue
        XX, YY = X[:, x0i:x1i], Y[y0i:y1i, :]
        st[y0i:y1i, x0i:x1i] += np.exp(-((XX - cx) / sw) ** 2) * np.exp(-(YY - ys) / L) * rr.uniform(0.5, 1.0)
    lum = lum * (1 - 0.10 * np.clip(st * (0.7 + 0.5 * noise_cols), 0, 1.4))
    # 天光：楼顶附近亮一点，往下渐暗
    amb = 0.80 + 0.32 * np.exp(-(Y - b.top) / 380)
    tone = np.array(b.tone, np.float32)
    return tone[None, None, :] * (lum * amb)[..., None]


def interior(win, X, Y):
    lx, ly, rad = win.lamp
    g = np.exp(-((X - lx) ** 2 + (Y - ly) ** 2) / (2 * rad * rad))
    return win.I * (0.11 + 0.89 * g)


PANE = {0: 0.50, 1: 0.34, 2: 0.155, 3: 0.135}
TEXT_GAIN = {0: 0.85, 1: 1.05, 2: 1.55, 3: 1.55}
CURTAIN = {0: 0.52, 1: 0.52, 2: 0.22, 3: 0.21}
TEXT_GLOW = ((1.2, 0.26), (3.5, 0.05))       # 只对字做的小半径辉光：(sigma, 权重)


def draw_window(col, emit, ox, oy, win, wall_tone, txt, sky_col, temit=None):
    f = win.frame
    spill = 0 if win.layer < 2 else (150 if win.layer == 3 else 50)
    mg = 14 if win.layer < 2 else int(3 * (0.30 * win.gw + 14) - 0.2 * win.gw)
    x0 = int(floor(win.gx - f - mg)) - ox
    x1 = int(ceil(win.gx + win.gw + f + mg)) - ox
    y0 = int(floor(win.gy - f - max(6, mg))) - oy
    y1 = int(ceil(win.gy + win.gh + f + win.sill + max(spill, mg) + 2)) - oy
    hh, ww = col.shape[:2]
    x0, x1, y0, y1 = max(0, x0), min(ww, x1), max(0, y0), min(hh, y1)
    if x1 <= x0 or y1 <= y0:
        return
    lw, lh = x1 - x0, y1 - y0
    ax, ay = ox + x0, oy + y0
    X = np.arange(lw, dtype=np.float32)[None, :] + ax + 0.5
    Y = np.arange(lh, dtype=np.float32)[:, None] + ay + 0.5
    c = col[y0:y1, x0:x1]
    e = emit[y0:y1, x0:x1]
    if win.kind == "arch":
        glass = cov_arch(win.gx, win.gy, win.gw, win.gh, ax, ay, lw, lh)
        outer = cov_arch(win.gx - f, win.gy - f, win.gw + 2 * f, win.gh + f, ax, ay, lw, lh, grow=0)
    else:
        glass = cov_rect(win.gx, win.gy, win.gx + win.gw, win.gy + win.gh, ax, ay, lw, lh)
        outer = cov_rect(win.gx - f, win.gy - f, win.gx + win.gw + f, win.gy + win.gh + f, ax, ay, lw, lh)
    tone = np.array(wall_tone, np.float32)
    frame_col = tone * ((0.55 if win.lit else 0.82) if win.layer == 3 else (0.7 if win.lit else 0.9))
    c[:] = c * (1 - outer[..., None]) + frame_col * outer[..., None]
    bars = np.zeros_like(glass)
    if win.kind == "transom":
        ty = win.gy + win.gh * 0.24
        bars = np.maximum(bars, cov_rect(win.gx, ty, win.gx + win.gw, ty + 6, ax, ay, lw, lh))
        bars = np.maximum(bars, cov_rect(win.gx + win.gw / 2 - 2.5, win.gy, win.gx + win.gw / 2 + 2.5, ty, ax, ay, lw, lh))
    pane = glass * (1 - bars)
    if win.lit:
        inten = interior(win, X, Y)
        reveal = 0.80 + 0.20 * smoothstep(win.gy, win.gy + 14, Y) * smoothstep(win.gx - 2, win.gx + 10, X)
        em = inten * PANE[win.layer] * reveal
        cl, cr = win.curtain
        if cl or cr:
            cur = np.zeros_like(glass)
            if cl:
                cur = np.maximum(cur, 1 - smoothstep(win.gx + win.gw * cl - 1.5, win.gx + win.gw * cl + 1.5, X))
            if cr:
                cur = np.maximum(cur, smoothstep(win.gx + win.gw * (1 - cr) - 1.5, win.gx + win.gw * (1 - cr) + 1.5, X))
            folds = 0.72 + 0.28 * np.cos((X - win.gx) / (3.2 + win.gw * 0.02) * 1.0) ** 2
            em = em * (1 - cur) + inten * CURTAIN[win.layer] * folds * cur
        t = txt[ay:ay + lh, ax:ax + lw]
        if win.layer >= 2:     # 字自己就是光：亮度只部分跟着灯位走
            lx, ly, rad = win.lamp
            g = np.exp(-((X - lx) ** 2 + (Y - ly) ** 2) / (2 * rad * rad))
            te = min(win.I, 1.25) * (0.48 + 0.52 * g) * TEXT_GAIN[win.layer] * t * reveal
        else:
            te = inten * TEXT_GAIN[win.layer] * t * reveal
        c[:] = c * (1 - pane[..., None])
        e += (em + te) * pane
        if temit is not None:
            temit[y0:y1, x0:x1] += te * pane
        if win.layer >= 2:
            # 窗台上那一溜光，和窗台下立面上的一小片暖色
            ys = win.gy + win.gh + f
            sl = cov_rect(win.gx - 6, ys, win.gx + win.gw + 6, ys + 3, ax, ay, lw, lh)
            e += sl * win.I * 0.20
            if spill:
                under = (Y > ys + win.sill) * np.exp(-(Y - ys - win.sill) / (spill * 0.32))
                horiz = np.exp(-((X - (win.gx + win.gw / 2)) / (win.gw * 0.62)) ** 2)
                e += under * horiz * win.I * (0.085 if win.layer == 3 else 0.05) * (1 - outer)
                # 窗洞四周墙面被屋里的光照到的一圈
                dx = np.maximum(0, np.abs(X - (win.gx + win.gw / 2)) - win.gw * 0.30)
                dy = np.maximum(0, np.abs(Y - (win.gy + win.gh / 2)) - win.gh * 0.30)
                halo = np.exp(-(dx * dx + dy * dy) / (2 * (0.30 * win.gw + 14) ** 2))
                e += halo * (1 - outer) * win.I * 0.05
    else:
        # 暗窗：冷的玻璃，上半截映一点天光，斜着一道淡淡的反光
        refl = 1 - smoothstep(win.gy, win.gy + win.gh, Y)
        streak = np.exp(-(((X - win.gx) - (Y - win.gy) * 0.6 - win.gw * 0.25) / (win.gw * 0.18)) ** 2)
        dark = tone[None, None, :] * (0.62 + 0.22 * refl[..., None])             + sky_col[None, None, :] * (0.03 + 0.10 * refl[..., None] + 0.04 * streak[..., None])
        if win.closed:     # 拉上的窗帘：布的褶子，冷而暗
            folds = 0.75 + 0.25 * np.cos((X - win.gx) / 5.0) ** 2
            dark = (tone * 0.75)[None, None, :] * folds[..., None] + dark * 0.25
        c[:] = c * (1 - pane[..., None]) + dark * pane[..., None]
    if win.sill > 0:
        ys = win.gy + win.gh + f
        sc = cov_rect(win.gx - f - 7, ys, win.gx + win.gw + f + 7, ys + win.sill, ax, ay, lw, lh)
        top_hi = cov_rect(win.gx - f - 7, ys, win.gx + win.gw + f + 7, ys + 1.5, ax, ay, lw, lh)
        sill_col = tone * (1.22 if win.lit else 1.10)
        c[:] = c * (1 - sc[..., None]) + sill_col * sc[..., None]
        c += (top_hi * (0.5 if win.lit else 0.25))[..., None] * tone
        sh = cov_rect(win.gx - f - 7, ys + win.sill, win.gx + win.gw + f + 7, ys + win.sill + 5, ax, ay, lw, lh)
        c *= (1 - 0.35 * sh)[..., None]


def draw_parts(b, col, emit, ox, oy, lit_lookup):
    hh, ww = col.shape[:2]
    tone = np.array(b.tone, np.float32)
    for kind, x0, y0, x1, y1, extra in b.parts:
        ix0, ix1 = max(0, int(floor(x0 - 8)) - ox), min(ww, int(ceil(x1 + 8)) - ox)
        iy0, iy1 = max(0, int(floor(y0 - 30)) - oy), min(hh, int(ceil(y1 + 12)) - oy)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        lw, lh = ix1 - ix0, iy1 - iy0
        ax, ay = ox + ix0, oy + iy0
        c = col[iy0:iy1, ix0:ix1]
        e = emit[iy0:iy1, ix0:ix1]
        X = np.arange(lw, dtype=np.float32)[None, :] + ax + 0.5
        Y = np.arange(lh, dtype=np.float32)[:, None] + ay + 0.5
        cov = cov_rect(x0, y0, x1, y1, ax, ay, lw, lh)
        e *= (1 - cov)
        if kind in ("course", "band"):
            colr = tone * (1.09 if kind == "band" else 1.07)
            c[:] = c * (1 - cov[..., None]) + colr * cov[..., None]
            hi = cov_rect(x0, y0, x1, y0 + 1.5, ax, ay, lw, lh)
            c += (hi * 0.22)[..., None] * tone
            sh = cov_rect(x0, y1, x1, y1 + 6, ax, ay, lw, lh)
            c *= (1 - 0.18 * sh * smoothstep(y1 + 6, y1, Y))[..., None]
        elif kind == "pilaster":
            colr = tone * 1.10
            c[:] = c * (1 - cov[..., None]) + colr * cov[..., None]
            hi = cov_rect(x0, y0, x0 + 1.5, y1, ax, ay, lw, lh)
            c += (hi * 0.25)[..., None] * tone
            sh = cov_rect(x1, y0, x1 + 7, y1, ax, ay, lw, lh) * smoothstep(x1 + 7, x1, X)
            c *= (1 - 0.30 * sh)[..., None]
        elif kind == "pipe":
            colr = tone * 0.85
            c[:] = c * (1 - cov[..., None]) + colr * cov[..., None]
            hi = cov_rect(x0 + 1.5, y0, x0 + 3, y1, ax, ay, lw, lh)
            c += (hi * 0.5)[..., None] * tone
            sh = cov_rect(x1, y0, x1 + 4, y1, ax, ay, lw, lh)
            c *= (1 - 0.35 * sh)[..., None]
            for yy in np.arange(y0 + 90, y1, 230.0):
                br = cov_rect(x0 - 3, yy, x1 + 3, yy + 5, ax, ay, lw, lh)
                c[:] = c * (1 - br[..., None]) + (tone * 0.7) * br[..., None]
        elif kind == "slab":
            colr = tone * 1.22
            c[:] = c * (1 - cov[..., None]) + colr * cov[..., None]
            hi = cov_rect(x0, y0, x1, y0 + 1.5, ax, ay, lw, lh)
            c += (hi * 0.4)[..., None] * tone
            sh = cov_rect(x0, y1, x1, y1 + 9, ax, ay, lw, lh) * smoothstep(y1 + 9, y1, Y)
            c *= (1 - 0.45 * sh)[..., None]
            # 楼板底面被下面一层亮着的门照暖
            for wn in lit_lookup.get((b.bid, "below", round(x0)), []):
                if wn.lit:
                    g = cov * smoothstep(y0, y1, Y) * np.exp(-((X - (wn.gx + wn.gw / 2)) / (wn.gw * 0.9)) ** 2)
                    e += g * wn.I * 0.10
        elif kind == "parapet":
            dwin, wwin, plants = extra
            colr = tone * 0.98
            n = 1 + 0.05 * fbm(lh, lw, int(x0 + y0) % 9973, octaves=4, base=40)
            c[:] = c * (1 - cov[..., None]) + (colr[None, None, :] * n[..., None]) * cov[..., None]
            cop = cov_rect(x0 - 3, y0 - 6, x1 + 3, y0, ax, ay, lw, lh)
            e *= (1 - cop)
            c[:] = c * (1 - cop[..., None]) + (tone * 1.35) * cop[..., None]
            for wn in (dwin, wwin):
                if wn.lit:
                    g = cop * np.exp(-((X - (wn.gx + wn.gw / 2)) / (wn.gw * 0.75)) ** 2)
                    e += g * wn.I * 0.20
                    # 栏板上沿往下一点点的暖色（门里的光从栏板上方漫出来）
                    g2 = cov * np.exp(-(Y - y0) / 10.0) * np.exp(-((X - (wn.gx + wn.gw / 2)) / (wn.gw * 0.6)) ** 2)
                    e += g2 * wn.I * 0.03
            sh = cov_rect(x0, y0, x1, y0 + 7, ax, ay, lw, lh) * smoothstep(y0 + 7, y0, Y)
            c *= (1 - 0.25 * sh)[..., None]
            for px, sz, kind_p, sd in plants:
                draw_plant(col, emit, ox, oy, px, y0 - 6, sz, kind_p, sd, tone)


def draw_plant(col, emit, ox, oy, px, base, sz, kind, sd, tone):
    """栏板上的一盆植物：盆 + 叶子的剪影，挡住后面门里的光。"""
    r = np.random.default_rng(sd)
    R = int(70 * sz)
    x0, y0 = int(px - R), int(base - R - 10)
    w, h = 2 * R, R + 14
    hh, ww = col.shape[:2]
    lx0, ly0 = x0 - ox, y0 - oy
    if lx0 < 0 or ly0 < 0 or lx0 + w > ww or ly0 + h > hh:
        return
    surf = skia.Surface(w, h)
    cv = surf.getCanvas()
    cv.clear(skia.ColorBLACK)
    cv.translate(-x0, -y0)
    pw, ph = r.uniform(18, 28) * sz, r.uniform(16, 24) * sz
    r_poly(cv, None, [(px - pw / 2, base - ph), (px + pw / 2, base - ph), (px + pw * 0.38, base), (px - pw * 0.38, base)])
    r_rect(cv, None, px - pw / 2 - 2, base - ph - 3, px + pw / 2 + 2, base - ph)
    top = base - ph
    if kind == "bush":
        for _ in range(int(r.integers(14, 24))):
            a = r.uniform(-np.pi * 0.95, -np.pi * 0.05)
            d = r.uniform(4, 30) * sz
            cx, cy = px + np.cos(a) * d * 1.1, top - 4 + np.sin(a) * d
            cv.save()
            cv.translate(cx, cy)
            cv.rotate(float(np.degrees(a) + r.uniform(-40, 40)))
            L = r.uniform(7, 13) * sz
            cv.drawOval(skia.Rect.MakeLTRB(-L, -L * 0.42, L, L * 0.42), _paint())
            cv.restore()
            r_line(cv, None, px, top, cx, cy, 1.0)
    elif kind == "spiky":
        for _ in range(int(r.integers(5, 9))):
            a = r.uniform(-np.pi * 0.78, -np.pi * 0.22)
            L = r.uniform(26, 52) * sz
            tip = (px + np.cos(a) * L, top + np.sin(a) * L)
            nx, ny = -np.sin(a) * 3.2 * sz, np.cos(a) * 3.2 * sz
            bx = px + r.uniform(-6, 6) * sz
            r_poly(cv, None, [(bx - nx, top + 1 - ny), (bx + nx, top + 1 + ny), tip])
    else:   # 垂下来的藤
        for _ in range(int(r.integers(4, 7))):
            xx = px + r.uniform(-pw * 0.5, pw * 0.5)
            L = r.uniform(10, 30) * sz
            cv.save()
            pth = skia.Path()
            pth.moveTo(xx, top)
            pth.quadTo(xx + r.uniform(-14, 14) * sz, top - L * 0.6, xx + r.uniform(-24, 24) * sz, top - L)
            p_ = _paint(1.4)
            cv.drawPath(pth, p_)
            cv.restore()
            for q in range(int(r.integers(3, 6))):
                t = r.uniform(0.2, 1.0)
                cx = xx + r.uniform(-18, 18) * sz * t
                cy = top - L * t
                cv.drawCircle(cx, cy, r.uniform(2.2, 4.0) * sz, _paint())
    m = surf.makeImageSnapshot().toarray()[..., 0].astype(np.float32) / 255.0
    c = col[ly0:ly0 + h, lx0:lx0 + w]
    c[:] = c * (1 - m[..., None]) + (tone * 0.55) * m[..., None]
    emit[ly0:ly0 + h, lx0:lx0 + w] *= (1 - m)


def draw_ac_units(b, col, emit, ox, oy):
    """F3：有些窗下挂着空调外机。"""
    if b.style != "stucco":
        return
    r = np.random.default_rng(b.seed + 21)
    hh, ww = col.shape[:2]
    tone = np.array(b.tone, np.float32)
    for win in b.wins:
        if r.random() > 0.55:
            continue
        side = r.choice([-1, 1])
        x0 = win.gx + win.gw / 2 + side * r.uniform(0, 26) - 40
        y0 = win.gy + win.gh + win.frame + win.sill + r.uniform(16, 30)
        x1, y1 = x0 + 80, y0 + 48
        ix0, ix1 = max(0, int(x0 - 4) - ox), min(ww, int(x1 + 10) - ox)
        iy0, iy1 = max(0, int(y0 - 4) - oy), min(hh, int(y1 + 14) - oy)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        lw, lh = ix1 - ix0, iy1 - iy0
        ax, ay = ox + ix0, oy + iy0
        X = np.arange(lw, dtype=np.float32)[None, :] + ax + 0.5
        Y = np.arange(lh, dtype=np.float32)[:, None] + ay + 0.5
        c = col[iy0:iy1, ix0:ix1]
        sh = cov_rect(x0 + 4, y1, x1 + 6, y1 + 10, ax, ay, lw, lh) * smoothstep(y1 + 10, y1, Y)
        c *= (1 - 0.4 * sh)[..., None]
        cov = cov_rect(x0, y0, x1, y1, ax, ay, lw, lh)
        body = tone[None, None, :] * (1.25 * (1 - 0.10 * ((Y - y0) / 48)))[..., None] * np.ones((1, lw, 1), np.float32)
        grille = 1 - 0.18 * (np.sin((Y - y0) * np.pi / 2.5) > 0.3) * cov_rect(x0 + 6, y0 + 6, x0 + 46, y1 - 6, ax, ay, lw, lh)
        fan = np.sqrt((X - (x0 + 58)) ** 2 + (Y - (y0 + 24)) ** 2)
        fanm = np.clip(17 - fan, 0, 1) * (1 - 0.5 * np.clip(13 - fan, 0, 1))
        cc = body * grille[..., None] * (1 - 0.35 * fanm)[..., None]
        c[:] = c * (1 - cov[..., None]) + cc * cov[..., None]
        emit[iy0:iy1, ix0:ix1] *= (1 - cov)
        hi = cov_rect(x0, y0, x1, y0 + 1.5, ax, ay, lw, lh)
        c += (hi * 0.5)[..., None] * tone
        for bx in (x0 + 8, x1 - 12):     # 托架
            br = cov_rect(bx, y1, bx + 4, y1 + 9, ax, ay, lw, lh)
            c[:] = c * (1 - br[..., None]) + (tone * 0.7) * br[..., None]


SHADOW = {0: (6, 0.18), 1: (9, 0.28), 2: (12, 0.34), 3: (16, 0.42)}
FOG = {0: 0.42, 1: 0.62, 2: 0.80, 3: 1.0}


def render_building(canvas, b, txt, sky, lit_lookup, tglow=None):
    pad = 90
    tops = [b.top] + [a[1] - 400 for n, a in b.roof if n in ("tank", "antenna", "mast", "dish", "box", "chimney")]
    ox = max(0, int(floor(b.x0 - pad)))
    x1 = min(W, int(ceil(b.x1 + pad)))
    oy = max(0, int(floor(min(tops) - pad)))
    w, h = x1 - ox, H - oy
    sil, roofm, detail = draw_sil(b, ox, oy, w, h)
    # 前面的楼在后面的楼上投一圈柔和的影子（纸剧场的层次）
    sg, sk = SHADOW[b.layer]
    sh = shift(gaussian(sil, sg), int(sg * 0.4), int(sg * 0.15))
    canvas[oy:H, ox:x1] *= (1 - sk * sh)[..., None]
    col = facade(b, ox, oy, w, h)
    emit = np.zeros((h, w), np.float32)
    temit = np.zeros((h, w), np.float32) if b.layer >= 2 else None
    sky_ref = sky[max(0, int(b.top) - 300), int(np.clip((b.x0 + b.x1) / 2, 0, W - 1))]
    # 楼顶附属物：比墙更暗的剪影，箍和梯子略亮
    tone = np.array(b.tone, np.float32)
    col = col * (1 - roofm[..., None]) + (tone * 0.72) * roofm[..., None]
    col = col + (detail * roofm * 0.35)[..., None] * tone
    for win in b.wins:
        if win.gy > H or win.gx > x1 or win.gx + win.gw < ox:
            continue
        draw_window(col, emit, ox, oy, win, b.tone, txt, sky_ref, temit)
    draw_parts(b, col, emit, ox, oy, lit_lookup)
    draw_ac_units(b, col, emit, ox, oy)
    # 天光勾出的上沿
    rim = np.clip(sil - shift(sil, 2, 0), 0, 1)
    col = col + (rim * 0.5)[..., None] * (sky_ref * 0.9 + tone * 0.6)
    col = col + emit[..., None] * LAMP
    # 雾：越远越重，越低越重
    Y = np.arange(h, dtype=np.float32)[:, None] + oy
    T = FOG[b.layer] - (0.10 if b.layer < 2 else 0.03 if b.layer == 2 else 0.0) * smoothstep(1000, 1700, Y)
    haze = sky[oy:H, ox:x1] * 0.92
    col = col * T[..., None] + haze * (1 - T[..., None])
    canvas[oy:H, ox:x1] = canvas[oy:H, ox:x1] * (1 - sil[..., None]) + col * sil[..., None]
    if tglow is not None:
        tg = tglow[oy:H, ox:x1]
        tg *= (1 - sil)
        if temit is not None:
            tg += temit * T * sil


def make_sky():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.clip(yy / 1400, 0, 1)
    sky = lerp(rgbf("#04070f"), rgbf("#111a2e"), (t ** 1.25)[..., None])
    sky = sky + (np.exp(-((yy - 1380) / 1000.0) ** 2) * 0.030)[..., None] * rgbf("#6f6a88")
    glow = np.exp(-((yy - 1380) / 360.0) ** 2) * (0.85 + 0.15 * np.cos(xx / W * 3.1 - 0.4))
    sky = sky + (glow * 0.13)[..., None] * rgbf("#6f6a88")
    sky = sky + (np.exp(-((yy - 1330) / 130.0) ** 2) * 0.05)[..., None] * rgbf("#8f88a4")
    # 被城市从下面照亮的薄云：横向拉长，下缘更亮
    n = fbm(H, W // 4, SEED + 31, octaves=6, base=150)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    band = smoothstep(40, 420, yy) * (1 - smoothstep(1150, 1450, yy))
    cl = smoothstep(0.0, 0.7, n) * band
    sky = sky + (cl * (0.036 + 0.12 * t ** 1.1))[..., None] * rgbf("#8a86a8")
    return sky


def check_overflow(wins, txt):
    """每扇亮窗：字的墨迹不能出文字区，也不能贴着 foreignObject 的下沿（那说明被裁了）。"""
    bad = []
    for w in wins:
        if not w.lit or w.ask < 0:
            continue
        x, y, bw, bh = w.box
        y += w.off
        X0, X1 = int(floor(w.gx)), int(ceil(w.gx + w.gw))
        Y0, Y1 = int(floor(w.gy)), int(ceil(w.gy + w.gh))
        m = txt[Y0:Y1, X0:X1] > 0.08
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            bad.append((ASKS[w.ask][1], "没有字"))
            continue
        ex0, ex1, ey0, ey1 = xs.min() + X0, xs.max() + X0 + 1, ys.min() + Y0, ys.max() + Y0 + 1
        if ex0 < x - 1.5 or ex1 > x + bw + 1.5 or ey1 > y + bh + 1.5 or ey1 >= y + bh + 3:
            bad.append((ASKS[w.ask][1], f"字 [{ex0},{ey0}-{ex1},{ey1}] 框 [{x:.0f},{y:.0f}-{x + bw:.0f},{y + bh:.0f}]"))
    print(f"  出框检查：{len(bad)} 处" + ("" if not bad else "：" + "; ".join(f"{a} {b}" for a, b in bad[:8])))


def hue_check(img, mask):
    px = img[mask > 0.5]
    if len(px) == 0:
        return
    s = px.sum(1)
    sel = px[s > np.quantile(s, 0.5)]
    rat = (sel / sel[:, :1]).mean(0)
    print(f"  亮窗色比 R:G:B = 1 : {rat[1]:.3f} : {rat[2]:.3f}（灯色 1 : {LAMP[1] / LAMP[0]:.3f} : {LAMP[2] / LAMP[0]:.3f}）")


def warm_bloom(img, thr, sigmas, weights):
    """按最大通道取阈值、整体按比例缩，辉光的色相和光源一样。"""
    m = img.max(axis=2, keepdims=True)
    src = img * np.clip((m - thr) / np.maximum(m, 1e-6), 0, None)
    acc = np.zeros_like(img)
    for s_, w_ in zip(sigmas, weights):
        acc += w_ * gaussian(src, s_)
    return img + acc


def tonemap(img):
    m = img.max(axis=2, keepdims=True)
    k = 0.62
    f = np.where(m <= k, m, k + (1 - k) * (1 - np.exp(-(m - k) / (1 - k))))
    return img * (f / np.maximum(m, 1e-6))


def main():
    front = front_buildings()
    near = near_buildings()
    mid = mid_buildings()
    far = far_buildings()
    all_b = far + mid + near + front

    occ_front = union_mask(front)
    occ_near = np.maximum(occ_front, union_mask(near))
    occ_mid = np.maximum(occ_near, union_mask(mid))
    set_visibility([w for b in near for w in b.wins], occ_front)
    set_visibility([w for b in mid for w in b.wins], occ_near)
    set_visibility([w for b in far for w in b.wins], occ_mid)
    front_w = [w for b in front for w in b.wins if w.lit]
    near_w = [w for b in near for w in b.wins if w.lit and w.vis >= 0.97]
    for b in near:      # 被前排挡住一部分的近排亮窗：不放问话，关灯
        for w in b.wins:
            if w.lit and w.vis < 0.97:
                w.lit = False
    used = assign_texts(front_w, near_w)

    lit_lookup = {}
    for b in front:
        for kind, x0, y0, x1, y1, extra in b.parts:
            if kind == "slab":
                # 这块楼板正下方那一层的门窗
                below = [w for w in b.wins if x0 <= w.gx + w.gw / 2 <= x1 and w.kind == "door"
                         and 0 < w.gy - y1 < 60]
                lit_lookup[(b.bid, "below", round(x0))] = below

    print(f"  [{time.time() - T0:.1f}s] 布局完成，开始排字")
    win_txt = svg_mask(build_window_text(front_w + near_w), name="03_windows_text", css=CSS)
    mid_txt = svg_mask(build_texture_text([w for b in mid for w in b.wins], 3.0, 1.32, SEED + 5), name="03_mid_text")
    star_svg, star_pts = build_stars()
    far_txt = svg_mask(build_texture_text([w for b in far for w in b.wins], 2.15, 1.30, SEED + 6) + star_svg,
                       name="03_far_text")
    print(f"  [{time.time() - T0:.1f}s] 字排完，开始合成")
    check_overflow(front_w + near_w, win_txt)

    sky = make_sky()
    canvas = sky.copy()
    # 星：几句很小的回答
    starm = np.zeros((H, W), np.float32)
    for x, y, sz, op in star_pts:
        y0, y1, x0, x1 = int(y - 3 * sz), int(y + 2 * sz), int(x - 30 * sz), int(x + 30 * sz)
        starm[y0:y1, x0:x1] = far_txt[y0:y1, x0:x1]
    canvas = canvas + starm[..., None] * rgbf("#dfe6f5") * 0.9
    for b in far:
        render_building(canvas, b, far_txt, sky, lit_lookup)
    # 远排脚下的雾
    Y = np.arange(H, dtype=np.float32)[:, None]
    mist1 = smoothstep(1050, 1450, Y) * (0.75 + 0.25 * fbm(H, W, SEED + 41, octaves=4, base=260))
    canvas = canvas + (mist1 * 0.045)[..., None] * rgbf("#5d6683")
    for b in mid:
        render_building(canvas, b, mid_txt, sky, lit_lookup)
    mist2 = smoothstep(1250, 1650, Y) * (0.7 + 0.3 * fbm(H, W, SEED + 42, octaves=4, base=300))
    canvas = canvas + (mist2 * 0.035)[..., None] * rgbf("#5d6683")
    tglow = np.zeros((H, W), np.float32)
    for b in near:
        render_building(canvas, b, win_txt, sky, lit_lookup, tglow)
    for b in front:
        render_building(canvas, b, win_txt, sky, lit_lookup, tglow)
    glow = sum(wt * gaussian(tglow, sg) for sg, wt in TEXT_GLOW)
    canvas = canvas + glow[..., None] * LAMP
    print(f"  [{time.time() - T0:.1f}s] 楼画完")

    # 色相检查：前排亮窗里的像素
    lm = np.zeros((H, W), np.float32)
    for w in front_w:
        if w.lit:
            lm[int(w.gy) + 4:int(w.gy + w.gh) - 4, int(w.gx) + 4:int(w.gx + w.gw) - 4] = 1
    hue_check(canvas, lm)

    img = warm_bloom(canvas, thr=0.12, sigmas=(2.5, 7, 20, 56, 150), weights=(0.05, 0.06, 0.25, 0.28, 0.18))
    img = img * vignette(H, W, strength=0.30, cy=0.52)[..., None]
    img = tonemap(img * 1.08)      # 整体曝光提一点：手机信息流里偏暗的图吃亏
    grain = paper(H, W, (1.0, 1.0, 1.0), seed=SEED + 3, mottle=0.05, fiber=0.6, grain=0.02)
    img = img * (0.94 + 0.06 * grain)
    img = np.clip(img * 1.04, 0, 1)
    hue_check(img, lm)
    save(img, OUT, seed=SEED)
    print(f"  总耗时 {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
