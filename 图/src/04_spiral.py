"""图 4「螺」：一场对话从中心的「你好。」逐字螺旋写到外圈的「再见。」，右侧正被风吹成墨屑。

做法：先按对话总字数解出螺线（字号和圈距随半径长，对整体缩放二分，让最后一个字正好落在外圈），
每个字一个 <text>，两种墨用 R/G 两个通道一次排出。风是不定常的流函数场（主流向右上 + 随流漂移的旋涡），
原位的字按风蚀程度在 numpy 里蚀孔、起毛；脱离的字沿风场积分出新位置，边飞边转、变小、变淡；
它们的蒙版再切成大小两级的 Voronoi 碎块，每块在某个时刻离开字身（下风侧先掉）、沿同一风场漂走、越远越碎越淡；
每个正在散的字还一阵阵往风里掉墨粉（流迹线），所以粉成簇、顺风、遇旋涡卷起。纸、光在最后乘上去。

    python -X utf8 04_spiral.py          # 出图 ../04-螺.png
    python -X utf8 04_spiral.py --dust   # 只算粉尘，快速看风场（work/04_debug_dust.png）
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).parent))
from common import *          # noqa
from texts_spiral import CONVERSATION, NOTE

SEED = 4
rng = np.random.default_rng(SEED)

CX, CY = 1080.0, 2000.0
R_OUT = 760.0               # 最后一个字的中心半径
S_IN, S_OUT = 23.5, 59.0    # 字号（缩放前的比例；实际由二分解出）
R_IN = 2.3                  # 起点半径 = R_IN × 首字字号
TRACK = 1.10                # 字距（em）
PITCH = 1.38                # 圈距 = PITCH × 字号
GAMMA = 0.55                # 字号沿字序的增长曲线
END_ANGLE = np.deg2rad(186.0)   # 最后一个字落在 9 点钟略偏上

INK_U = rgbf("#1d1a17")     # 对方：黑墨
INK_C = rgbf("#2c3a4f")     # 我：深蓝灰墨
PAPER = (0xf1, 0xeb, 0xdf)
PENCIL = rgbf("#6b6660")

NOTE_XY = (172, 3008)
NOTE_ROT = -2.2
NOTE_SIZE = 30


# ---------------------------------------------------------------- 螺线排字

def slots():
    """逐字展开对话；句与句之间空一格（None）。返回 [(字, who, 句序), ...]。"""
    out = []
    for k, (who, t) in enumerate(CONVERSATION):
        if k:
            out.append((None, who, k))
        for ch in t:
            out.append((ch, who, k))
    return out


def march(n, f):
    """按缩放因子 f 沿螺线逐格前进，返回每格 (θ, r, 切线角, 字号)。"""
    sizes = f * (S_IN + (S_OUT - S_IN) * (np.arange(n) / (n - 1)) ** GAMMA)
    th, r = 0.0, R_IN * sizes[0]
    res = []

    def walk(dist, s):
        nonlocal th, r
        b = PITCH * s / (2 * np.pi)
        steps = max(1, int(np.ceil(dist / 1.5)))
        ds = dist / steps
        for _ in range(steps):
            dth = ds / np.hypot(r, b)
            th += dth
            r += b * dth

    for i in range(n):
        s = sizes[i]
        adv = s * TRACK
        walk(adv / 2, s)
        b = PITCH * s / (2 * np.pi)
        tang = np.arctan2(b * np.sin(th) + r * np.cos(th), b * np.cos(th) - r * np.sin(th))
        res.append((th, r, tang, s))
        walk(adv / 2, s)
    return np.array(res)


def layout():
    sl = slots()
    n = len(sl)
    lo, hi = 0.3, 3.0
    for _ in range(40):
        f = (lo + hi) / 2
        if march(n, f)[-1, 1] < R_OUT:
            lo = f
        else:
            hi = f
    arr = march(n, (lo + hi) / 2)
    rot = END_ANGLE - arr[-1, 0]
    th = arr[:, 0] + rot
    chars = []
    for i, (ch, who, k) in enumerate(sl):
        x = CX + arr[i, 1] * np.cos(th[i])
        y = CY + arr[i, 1] * np.sin(th[i])
        chars.append(dict(i=i, ch=ch, who=who, utt=k, x=x, y=y, r=arr[i, 1], th=th[i],
                          ang=float(np.rad2deg(arr[i, 2] + rot)), s=arr[i, 3]))
    turns = (arr[-1, 0] - arr[0, 0]) / (2 * np.pi)
    print(f"  螺线：{n} 格，{turns:.2f} 圈，字号 {arr[0, 3]:.1f} → {arr[-1, 3]:.1f}px，"
          f"半径 {arr[0, 1]:.0f} → {arr[-1, 1]:.0f}px")
    return chars


# ---------------------------------------------------------------- 风

class Wind:
    """风从左下往右上吹，越靠左越陡（往上升），越靠右越平（往右飘），所以粉流在上半幅张成扇形。
    另有两个大而缓的旋涡随主流漂移（写在随流坐标 ξ = p − DRIFT·t 上的静态流函数里，无散度）。
    时间 t = 0 是「此刻」，t < 0 是过去。
    """
    G = 4.0
    X0, Y0 = -1600.0, -2400.0
    DRIFT = np.array([0.62, -0.74])

    # 此刻（t=0）的旋涡：(x, y, 半径, 边缘旋速)，旋速 > 0 屏幕上顺时针
    MAIN = [(1930, 980, 360, 0.55), (1250, 520, 300, -0.40)]

    def __init__(self, seed):
        gw, gh = int(6400 / self.G), int(6400 / self.G)
        yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float64)
        xx = xx * self.G + self.X0
        yy = yy * self.G + self.Y0
        psi = np.zeros_like(xx)
        pts = []
        for x_, y_, sig, spd in self.MAIN:
            pts.append((np.array([x_, y_], np.float64), sig, spd * sig / 0.6065))
        for p, sig, gam in pts:
            psi += gam * np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * sig * sig))
        self.vort = pts
        psi += 30.0 * fbm(gh, gw, seed + 1, octaves=3, base=160).astype(np.float64)
        gy, gx = np.gradient(psi, self.G)
        self.vx = gy.astype(np.float32)        # ∂ψ/∂y
        self.vy = (-gx).astype(np.float32)     # −∂ψ/∂x

    @staticmethod
    def base(p):
        x, y = p[:, 0], p[:, 1]
        phi = np.deg2rad(47 + 20 * np.tanh((1650 - x) / 650) + 6 * np.tanh((y - 1700) / 600))
        return np.stack([np.cos(phi), -np.sin(phi)], 1)

    def vel(self, p, t):
        q = p - self.DRIFT[None, :] * t[:, None]
        c = np.vstack([(q[:, 1] - self.Y0) / self.G, (q[:, 0] - self.X0) / self.G])
        v = np.stack([map_coordinates(self.vx, c, order=1, mode="nearest"),
                      map_coordinates(self.vy, c, order=1, mode="nearest")], 1)
        return self.base(p) + v

    def advect(self, p, t0, t1, speed=1.0, steps=32, keep=0):
        """粒子从时刻 t0 的位置 p 积分到 t1（逐点不同），RK2。speed 为逐点倍速。
        keep>0 时同时返回 keep 个等时距快照（最后一个即终点）。"""
        p = np.asarray(p, np.float64).copy()
        t = np.asarray(t0, np.float64) * np.ones(len(p))
        t1 = np.asarray(t1, np.float64) * np.ones(len(p))
        sp = (np.asarray(speed, np.float64) * np.ones(len(p)))[:, None]
        dt = (t1 - t) / steps
        hist = []
        every = max(1, steps // keep) if keep else 0
        for k in range(steps):
            v1 = self.vel(p, t)
            v2 = self.vel(p + 0.5 * dt[:, None] * sp * v1, t + 0.5 * dt)
            p = p + dt[:, None] * sp * v2
            t = t + dt
            if keep and (k + 1) % every == 0:
                hist.append(p.copy())
        return (p, np.array(hist)) if keep else p


# ---------------------------------------------------------------- 风蚀程度与飞行

D_ERODE, D_LOOSE, D_FLY = 0.06, 0.34, 0.45     # 完好 | 蚀孔起毛 | 松动 | 脱离
T_FLY = 1500.0              # τ = 1 的字已经飞了多久
LAM0, LAM1 = 0.06, 0.36     # 碎块离开字身的时刻（τ 单位）
CHAR_SPEED = 1.15           # 字身的飞行速度；掉下的碎块多半落在它后面


ERODE_AXIS = np.array([0.92, -0.39]) / np.hypot(0.92, 0.39)    # 螺旋上沿这个方向越靠前越散


def flight_time(tau):
    return T_FLY * np.power(np.clip(tau, 0, None), 1.35)


def disperse(chars):
    """每个字的风蚀程度 D∈[0,1]：x 越靠右越大，内圈有遮挡，边界用低频噪声弄得参差。"""
    nz = fbm(H, W, SEED + 31, octaves=4, base=260)
    nz2 = fbm(H, W, SEED + 32, octaves=3, base=60)
    for c in chars:
        u = ((c["x"] - CX) * ERODE_AXIS[0] + (c["y"] - CY) * ERODE_AXIS[1]) / R_OUT
        rn = c["r"] / R_OUT
        front = 0.04 + 0.34 * (1 - rn) ** 2
        xi, yi = int(np.clip(c["x"], 0, W - 1)), int(np.clip(c["y"], 0, H - 1))
        d = (u - front) / (0.84 - front) + 0.15 * nz[yi, xi] + 0.07 * nz2[yi, xi] + rng.normal(0, 0.06)
        if c["i"] < 4 or c["r"] < 0.18 * R_OUT:
            d = 0.0
        c["D"] = float(np.clip(d, 0, 1))
        c["tau"] = float(np.clip((c["D"] - D_FLY) / (1 - D_FLY), 0, 1)) if c["D"] > D_FLY else 0.0


def rot_at(c, t):
    return c["ang"] + c["spin"] * t


def sc_at(t):
    return 1.0 - 0.42 * smoothstep(0.0, 0.5, t)


def op_at(t):
    return 1.0 - 0.38 * smoothstep(0.0, 0.5, t)


def fly(chars, wind):
    flying = [c for c in chars if c["ch"] and c["D"] > D_FLY]
    for c in flying:
        c["spin"] = float(rng.uniform(90, 420) * rng.choice([-1, 1]))
        c["speed"] = float(CHAR_SPEED * rng.uniform(0.62, 1.4))
        c["tau"] = float(np.clip(c["tau"] + rng.normal(0, 0.07), 0.01, 1))
        c["lat"] = float(rng.normal(0, 1) * c["s"] * 1.6)

    def run(sub):
        o = np.array([(c["x"], c["y"]) for c in sub])
        t0 = -flight_time(np.array([c["tau"] for c in sub]))
        end, hist = wind.advect(o, t0, 0.0, speed=np.array([c["speed"] for c in sub]), steps=48, keep=48)
        nrm = np.array([WIND_DIR[1], -WIND_DIR[0]])
        for k, c in enumerate(sub):
            c["t0"] = float(t0[k])
            path = np.vstack([o[k:k + 1], hist[:, k]])            # 49 个等时距点：t0 … 0
            ramp = smoothstep(0.0, 1.0, np.linspace(0, 1, 49)) * min(1.0, c["tau"] / 0.3)
            c["path"] = path + ramp[:, None] * nrm[None, :] * c["lat"]
            c["pos"] = c["path"][-1]

    run(flying)
    # 避让：还看得见字身的飞字之间、以及和原位的字之间不许压字
    # 半径取 0.64 字号：字转了角，四角也要让开
    KR = 0.64
    fixed = [(c["x"], c["y"], KR * c["s"]) for c in chars if c["ch"] and c["D"] <= D_FLY]
    placed = []
    order = sorted([c for c in flying if c["tau"] < LAM1], key=lambda c: c["tau"])
    moved = gave_up = 0
    for c in order:
        tau0, lat0 = c["tau"], c["lat"]
        for attempt in range(28):
            rad = KR * c["s"] * sc_at(c["tau"])
            px, py = c["pos"]
            ok = all(np.hypot(px - x, py - y) > rad + rr for x, y, rr in fixed + placed)
            if ok:
                break
            if attempt == 27:                    # 实在让不开：这个字已经散尽，只剩碎屑
                c["tau"] = max(c["tau"], LAM1)
                gave_up += 1
                run([c])
                break
            k = attempt // 2 + 1
            c["tau"] = float(np.clip(tau0 + (1 if attempt % 2 else -1) * 0.014 * k, 0.005, LAM1 - 0.01))
            c["lat"] = lat0 + float(rng.normal(0, 1)) * c["s"] * 0.8 * min(k, 6)
            run([c])
            moved += 1
        if c["tau"] < LAM1:
            placed.append((c["pos"][0], c["pos"][1], KR * c["s"] * sc_at(c["tau"])))
    print(f"  脱离的字 {len(flying)} 个（字身尚在 {sum(c['tau'] < LAM1 for c in flying)} 个），"
          f"避让重算 {moved} 次，散尽 {gave_up} 个")
    return flying


def path_at(c, t):
    """字在时刻 t（t0 ≤ t ≤ 0）的位置。"""
    t = np.asarray(t, np.float64)
    f = np.clip((t - c["t0"]) / max(-c["t0"], 1e-6), 0, 1) * 48
    j = np.clip(np.floor(f).astype(int), 0, 47)
    a = (f - j)[..., None]
    return c["path"][j] * (1 - a) + c["path"][j + 1] * a


# ---------------------------------------------------------------- SVG

def svg_rgb(body, w=W, h=H, ss=2, name="rgb", defs=""):
    """和 common.svg_mask 一样排版，但保留 R/G/B 三个通道（各当一张蒙版用）。"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;padding:0;background:#000;overflow:hidden}}
svg{{display:block}}
text{{font-kerning:normal;text-rendering:geometricPrecision}}
</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" width="{w * ss}" height="{h * ss}" viewBox="0 0 {w} {h}">
<defs>{defs}</defs>
{body}
</svg></body></html>"""
    img = chrome_shot(html, w * ss, h * ss, name=name).astype(np.float32) / 255.0
    if ss != 1:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 1)


def glyph(ch, x, y, ang, s, fill):
    return (f'<text transform="translate({x:.2f},{y:.2f}) rotate({ang:.2f})" font-size="{s:.2f}" '
            f'text-anchor="middle" dominant-baseline="central" fill="{fill}">{esc(ch)}</text>')


FILL = {"u": "#f00", "c": "#0f0"}
WEIGHT = 500


def build_inplace(chars):
    """原位的字（完好 + 蚀孔 + 松动）；松动的字顺风挪一点、歪一点。笔记放在蓝通道。"""
    out = [f'<g font-family="{FONT["zh"]}" font-weight="{WEIGHT}">']
    for c in chars:
        if not c["ch"] or c["D"] > D_FLY:
            continue
        x, y, a = c["x"], c["y"], c["ang"]
        if c["D"] > D_LOOSE:
            k = (c["D"] - D_LOOSE) / (D_FLY - D_LOOSE)
            x += k * rng.uniform(0.5, 3.5)
            y -= k * rng.uniform(0.0, 2.5)
            a += k * rng.uniform(-7, 7)
        c["draw"] = (x, y, a)
        out.append(glyph(c["ch"], x, y, a, c["s"], FILL[c["who"]]))
    out.append("</g>")
    out.append(f'<text transform="translate({NOTE_XY[0]},{NOTE_XY[1]}) rotate({NOTE_ROT})" '
               f'font-family="STKaiti, KaiTi" font-size="{NOTE_SIZE}" fill="#00f">{esc(NOTE)}</text>')
    return "\n".join(out)


SP = 96                     # 精灵格子边长（px）
SP_COLS = 24


def build_sprites(flying):
    """脱离的字各画在精灵表的一格里（转角、字号按字身最后的样子），之后在 numpy 里搬到画面上。"""
    rows = int(np.ceil(len(flying) / SP_COLS))
    out = [f'<g font-family="{FONT["zh"]}" font-weight="{WEIGHT}">']
    for k, c in enumerate(flying):
        tr = min(c["tau"], LAM1)
        gx, gy = (k % SP_COLS) * SP + SP // 2, (k // SP_COLS) * SP + SP // 2
        if c["tau"] < LAM1:
            fx, fy = c["pos"][0] - np.floor(c["pos"][0]), c["pos"][1] - np.floor(c["pos"][1])
        else:
            fx = fy = 0.0
        c["sprite"] = (gx, gy, fx, fy, rot_at(c, tr), sc_at(tr))
        out.append(glyph(c["ch"], gx + fx, gy + fy, rot_at(c, tr), c["s"] * sc_at(tr), FILL[c["who"]]))
    out.append("</g>")
    return "\n".join(out), SP_COLS * SP, max(1, rows) * SP


# ---------------------------------------------------------------- 蚀、碎

def unoise(h, w, seed, base, octaves=3):
    """近似均匀分布在 [0,1] 的噪声（阈值 = 面积比例）。"""
    n = fbm(h, w, seed, octaves=octaves, base=base)
    n = n / (n.std() + 1e-6)
    return (0.5 * (1 + erf(n / np.sqrt(2)))).astype(np.float32)


def soft_keep(n, th, w):
    t = np.clip((n - th) / (2 * w) + 0.5, 0, 1)
    return t * t * (3 - 2 * t)


WIND_DIR = np.array([0.74, -0.67]) / np.hypot(0.74, 0.67)


def line_kernel(length, direction, decay):
    """一侧的线核：从上风方向把墨抹过来。"""
    k = np.zeros((2 * length + 1, 2 * length + 1), np.float32)
    for t in np.linspace(0, length, length * 4 + 1):
        x = int(round(length - direction[0] * t))
        y = int(round(length - direction[1] * t))
        k[y, x] = max(k[y, x], float(np.exp(-t / decay)))
    return k / k.sum()


def erode(ms, e, seed):
    """ms: [覆盖率, ...]（同一形状），e: 逐像素风蚀程度 [0,1]。蚀孔、起毛、顺风的毛丝。"""
    h, w = e.shape
    wx = fbm(h, w, seed + 1, octaves=2, base=3.0)
    wy = fbm(h, w, seed + 2, octaves=2, base=3.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp = 2.2 * e
    holes = unoise(h, w, seed + 4, 2.2, 2)
    bites = unoise(h, w, seed + 5, 7.0, 3)
    streak = cv2.filter2D(unoise(h, w, seed + 3, 1.6, 2), -1, line_kernel(6, WIND_DIR, 4.0))
    streak = np.clip((streak - 0.5) * 3.2 + 0.5, 0, 1)
    th = 0.58 * np.clip(e, 0, 1) ** 1.25
    keep = soft_keep(holes, th, 0.03 + 0.04 * e)
    tb = 0.55 * np.clip((e - 0.45) / 0.55, 0, 1) ** 1.2
    keep *= soft_keep(bites, tb, 0.06)
    kern = line_kernel(9, WIND_DIR, 3.5)
    out = []
    for m in ms:
        mw = cv2.remap(m, xx + wx * amp, yy + wy * amp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        smear = cv2.filter2D(mw, -1, kern)
        hair = smoothstep(0.18, 0.5, smear) * smoothstep(0.62, 0.85, streak) * (1 - mw)
        m2 = np.maximum(mw, hair * np.clip(e * 1.3, 0, 1) * 0.8)
        r = m2 * keep * (1 - 0.22 * e)
        out.append(np.where(e > 0, r, m))
    return out


def voronoi_cells(px, py, spacing, seed):
    """在点集所在范围铺一层抖动网格种子，返回 (每点的格子号, 到格界的距离, 种子坐标)。"""
    r = np.random.default_rng(seed)
    x0, x1, y0, y1 = px.min() - spacing, px.max() + spacing, py.min() - spacing, py.max() + spacing
    sx, sy = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
    seeds = np.stack([sx.ravel(), sy.ravel()], 1) + r.uniform(0, spacing, (sx.size, 2))
    d, ii = cKDTree(seeds).query(np.stack([px, py], 1), k=2)
    return ii[:, 0], 0.5 * (d[:, 1] - d[:, 0]), seeds


class Ink:
    """画面上的两种墨（u / c）和三档墨粉。碎块用 max 合并，墨粉累加。"""
    SIG = (0.38, 0.6, 0.95)
    PROB = (0.62, 0.30, 0.08)

    def __init__(self):
        self.acc = np.zeros((2, H * W), np.float32)
        self.dust = np.zeros((3, 2, H * W), np.float32)

    def put(self, x, y, vals):
        """x, y 整数像素坐标；vals (N, 2)。"""
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        idx = y[ok] * W + x[ok]
        for ch in range(2):
            np.maximum.at(self.acc[ch], idx, vals[ok, ch])

    def grains(self, x, y, amp, voice, cls):
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = x - x0, y - y0
        for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                            (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xx, yy = x0 + dx, y0 + dy
            ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            flat = (cls[ok] * 2 + voice[ok]) * (H * W) + yy[ok] * W + xx[ok]
            np.add.at(self.dust.reshape(-1), flat, (amp * wgt)[ok])

    def dust_cover(self):
        out = np.zeros((2, H, W), np.float32)
        for cls, s in enumerate(self.SIG):
            for ch in range(2):
                d = self.dust[cls, ch].reshape(H, W)
                out[ch] += gaussian(d, s) * (2 * np.pi * s * s) ** 0.85
        c = 1 - np.exp(-out)
        return np.clip((c - 0.015) / 0.75, 0, 1) ** 1.15


def emit_dust(ink, start, hist, end, mass, voice, alpha, r, rate):
    """每块碎屑身后拖一小段墨粉：集中在它刚走过的那一段，横向很窄，所以有方向、成簇。"""
    seg = np.concatenate([start[None], hist], 0)          # (K+1, N, 2)
    dist = np.linalg.norm(end - start, axis=1)
    n = r.poisson(np.clip(mass * np.sqrt(dist + 1) * rate, 0, 30))
    if n.sum() == 0:
        return
    who = np.repeat(np.arange(len(n)), n)
    back = np.minimum(r.exponential(1.0, len(who)) * (18 + 0.05 * dist[who]), dist[who])   # 往回多少像素
    f = 1 - back / np.maximum(dist[who], 1e-6)
    kf = f * (seg.shape[0] - 1)
    j = np.clip(np.floor(kf).astype(int), 0, seg.shape[0] - 2)
    a = (kf - j)[:, None]
    p = seg[j, who] * (1 - a) + seg[j + 1, who] * a
    tang = seg[j + 1, who] - seg[j, who]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    spread = 0.6 + 0.08 * back + 0.002 * dist[who]
    p = p + nrm * (r.normal(0, 1, len(who)) * spread)[:, None]
    amp = alpha[who] * r.uniform(0.35, 1.0, len(who)) * np.exp(-back / 140) * np.exp(-dist[who] / 2600)
    cls = r.choice(3, len(who), p=Ink.PROB)
    ink.grains(p[:, 0], p[:, 1], amp, voice[who], cls)



T_SMOKE = 1100.0            # 一个字开始被蚀到现在，往外冒墨粉冒了多久（D 越大越久）


def streak_dust(chars, flying, wind, ink, bursts=18, per_burst=26, a0=240.0):
    """每个正在散的字像一处烟源：从它被蚀开始，一阵一阵往风里掉墨粉（每阵从字上某一点掉下一小撮），
    每粒从掉下的时刻沿风场积分到此刻。一撮粉在路上被拉长成顺风的一簇，遇到旋涡就卷起来；近浓远淡。
    每粒画成顺着此刻风向的一小段（运动拖影），所以有方向。"""
    t0 = time.time()
    r = np.random.default_rng(SEED + 80)
    P, T0, A, V, SPD = [], [], [], [], []
    for c in chars:
        if not c["ch"] or c["D"] <= D_ERODE:
            continue
        lvl = np.clip((c["D"] - D_ERODE) / (1 - D_ERODE), 0, 1)
        nb = int(r.poisson(bursts * lvl ** 0.75 * (c["s"] / 40) ** 2))
        if nb == 0:
            continue
        age = lvl * T_SMOKE                                  # 冒了多久
        start = -age - (flight_time(c["tau"]) if c["D"] > D_FLY else 0.0)
        stop = 0.0
        if c["D"] > D_FLY and c["tau"] >= LAM1:
            stop = c["t0"] + flight_time(LAM1)               # 字身散尽就不再掉
        # 掉粉的时刻：大多是刚掉的（近处密），少数是早掉的（拖到远处、飘出画面）
        ages = np.where(r.random(nb) < 0.82, r.exponential(a0, nb), r.exponential(4 * a0, nb))
        tb = np.maximum(stop - ages, start) if stop > start else np.full(nb, stop)
        nper = r.poisson(per_burst, nb) + 3
        te = np.repeat(tb, nper) + r.normal(0, 5, nper.sum())
        te = np.minimum(te, -0.5)
        if c["D"] > D_FLY:
            base = np.where((te < c["t0"])[:, None], [[c["x"], c["y"]]], path_at(c, np.maximum(te, c["t0"])))
        else:
            base = np.tile([c["x"], c["y"]], (len(te), 1))
        a = np.deg2rad(c["ang"])
        lx = np.repeat(r.uniform(-0.4, 0.4, nb) * c["s"], nper) + r.normal(0, 1.2, len(te))
        ly = np.repeat(r.uniform(-0.4, 0.4, nb) * c["s"], nper) + r.normal(0, 1.2, len(te))
        pos = base + np.stack([lx * np.cos(a) - ly * np.sin(a), lx * np.sin(a) + ly * np.cos(a)], 1)
        sp = np.repeat(r.uniform(0.85, 1.35, nb), nper) * (1 + r.normal(0, 0.006, len(te)))
        P.append(pos); T0.append(te); SPD.append(sp)
        A.append(np.full(len(te), 0.6 + 0.4 * lvl)); V.append(np.full(len(te), 0 if c["who"] == "u" else 1))
    P, T0, A, V, sp = map(np.concatenate, (P, T0, A, V, SPD))
    end = wind.advect(P, T0, 0.0, speed=sp, steps=36)
    age = -T0
    end = end + r.normal(0, 1, end.shape) * (1.4 + 0.006 * age)[:, None]
    amp = A * r.uniform(0.3, 1.0, len(P)) * np.exp(-age / 800) * smoothstep(0, 25, age)
    cls = r.choice(3, len(P), p=Ink.PROB)
    # 运动拖影：沿此刻风向往回拉一小段，分 4 点撒
    v = wind.vel(end, np.zeros(len(end))) * sp[:, None]
    L = np.clip(np.linalg.norm(v, axis=1) * r.lognormal(np.log(1.1), 0.5, len(end)), 0.0, 3.2)
    vd = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
    ns = 3
    for k in range(ns):
        q = end - vd * (L * k / (ns - 1))[:, None]
        ink.grains(q[:, 0], q[:, 1], amp * (1.0 - 0.15 * k) * (1.6 / ns) / (1 + 0.15 * L), V.astype(np.int64), cls)
    print(f"  流迹墨粉 {len(P) / 1e3:.0f}k 粒  {time.time() - t0:.1f}s")


# ---------------------------------------------------------------- 主流程

def rotated_box(x, y, ang, side):
    a = np.deg2rad(ang)
    u = np.array([np.cos(a), np.sin(a)]) * side / 2
    v = np.array([-np.sin(a), np.cos(a)]) * side / 2
    pts = np.array([[x, y] + u + v, [x, y] + u - v, [x, y] - u - v, [x, y] - u + v])
    return np.round(pts * 16).astype(np.int32)


def inplace_layer(chars, A, wind, ink):
    """原位的字：按 D 逐像素蚀孔起毛；少数细碎块已经被风揭起，挪到下风处。返回留在原位的两张墨。"""
    t0 = time.time()
    e = np.zeros((H, W), np.float32)
    for c in chars:
        if c["ch"] and D_ERODE < c["D"] <= D_FLY:
            x, y, a = c["draw"]
            cv2.fillConvexPoly(e, rotated_box(x, y, a, c["s"] * 1.18),
                               float((c["D"] - D_ERODE) / (D_FLY - D_ERODE)), cv2.LINE_8, 4)
    e = gaussian(e, 1.0)
    ys, xs = np.nonzero(e > 0.002)
    y0, y1, x0, x1 = ys.min() - 16, ys.max() + 16, xs.min() - 16, xs.max() + 16
    sl = (slice(y0, y1), slice(x0, x1))
    mu, mc = A[..., 0].copy(), A[..., 1].copy()
    eu, ec = erode([mu[sl], mc[sl]], e[sl], SEED + 50)
    mu[sl], mc[sl] = eu, ec

    # 揭起的细碎块
    r = np.random.default_rng(SEED + 51)
    m = mu[sl] + mc[sl]
    py, px = np.nonzero((m > 0.004) & (e[sl] > 0.05))
    if len(px):
        cid, bd, seeds = voronoi_cells(px + 0.5, py + 0.5, 3.0, SEED + 52)
        ncell = len(seeds)
        lam = r.uniform(0.30, 1.05, ncell)
        elig = r.random(ncell) < 0.32
        ecell = np.zeros(ncell)
        np.maximum.at(ecell, cid, e[sl][py, px])
        move = elig & (ecell > lam)
        mv = move[cid]
        if mv.any():
            ft = (ecell - lam) * 150 * r.uniform(0.6, 1.5, ncell)
            cells = np.nonzero(move)[0]
            st = seeds[cells] + [x0, y0]
            end, hist = wind.advect(st, -ft[cells], 0.0, steps=10, keep=5)
            disp = np.zeros((ncell, 2))
            disp[cells] = end - st
            dist = np.linalg.norm(disp, axis=1)
            pxm, pym = px[mv], py[mv]
            d = np.round(disp[cid[mv]]).astype(np.int64)
            fade = (0.85 * np.exp(-dist[cid[mv]] / 160))[:, None]
            crack = smoothstep(0.15, 0.9, bd[mv])[:, None]
            vals = np.stack([mu[sl][pym, pxm], mc[sl][pym, pxm]], 1) * fade * crack
            ink.put(pxm + x0 + d[:, 0], pym + y0 + d[:, 1], vals)
            mass = np.zeros(ncell)
            np.add.at(mass, cid[mv], m[pym, pxm])
            voice = np.zeros(ncell, np.int64)
            uu = np.zeros(ncell)
            np.add.at(uu, cid[mv], mu[sl][pym, pxm])
            voice[uu < 0.5 * mass] = 1
            emit_dust(ink, st, hist, end, mass[cells], voice[cells], np.full(len(cells), 0.5), r, 0.03)
            sub_u, sub_c = mu[sl], mc[sl]
            sub_u[pym, pxm] = 0
            sub_c[pym, pxm] = 0
        print(f"  原位：蚀 {int((e > 0.01).sum() / 1e3)}k px，揭起细块 {int(move.sum())} 个  {time.time() - t0:.1f}s")
    return mu, mc


def flying_layer(flying, S, wind, ink):
    """脱离的字：字身（还没离开的碎块）放到飞行位置；离开的碎块从离开时刻沿风场漂到此刻。
    碎块先是整块的笔画残段（粗块，约 10 px），飞远了才碎成细块、再散成零星墨点；
    留存概率和浓淡都按飞行距离指数衰减，所以近处密、远处只剩少数小点。"""
    t0 = time.time()
    r = np.random.default_rng(SEED + 60)
    sh, sw = S.shape[:2]
    kmap = np.full((sh, sw), -1, np.int64)
    for k in range(len(flying)):
        gx, gy = (k % SP_COLS) * SP, (k // SP_COLS) * SP
        kmap[gy:gy + SP, gx:gx + SP] = k
    tau = np.array([c["tau"] for c in flying])
    e_body = 0.12 + 0.48 * smoothstep(0.0, 0.45, tau)
    e = np.where(kmap >= 0, e_body[np.clip(kmap, 0, None)], 0).astype(np.float32)
    su, sc_ = erode([S[..., 0], S[..., 1]], e, SEED + 61)
    m = su + sc_
    py, px = np.nonzero((m > 0.004) & (kmap >= 0))
    kp = kmap[py, px]
    w_ = m[py, px]
    # 两级碎块：粗块（笔画残段）整块离开、整块飞；细块只在飞远后才各自散开
    cid, cbd, cseeds = voronoi_cells(px + 0.5, py + 0.5, 10.0, SEED + 63)
    fid, fbd, fseeds = voronoi_cells(px + 0.5, py + 0.5, 3.2, SEED + 62)
    nc, nf = len(cseeds), len(fseeds)
    cw = np.bincount(cid, weights=w_, minlength=nc)
    ccx = np.bincount(cid, weights=w_ * (px + 0.5), minlength=nc) / np.maximum(cw, 1e-6)
    ccy = np.bincount(cid, weights=w_ * (py + 0.5), minlength=nc) / np.maximum(cw, 1e-6)
    ck = np.full(nc, -1, np.int64)
    ck[cid] = kp
    ckc = np.clip(ck, 0, None)
    # 下风那一侧先掉：按粗块在字里沿风向的位置排离开时刻
    gxy = np.array([[c["sprite"][0] + c["sprite"][2], c["sprite"][1] + c["sprite"][3]] for c in flying])
    sz = np.array([c["s"] * c["sprite"][5] for c in flying])
    proj = ((ccx - gxy[ckc, 0]) * WIND_DIR[0] + (ccy - gxy[ckc, 1]) * WIND_DIR[1]) / sz[ckc]
    order = np.clip(0.5 - 0.75 * proj + 0.45 * r.uniform(-0.5, 0.5, nc), 0, 1)
    lamC = np.clip(LAM0 + (LAM1 - LAM0) * order, 0.02, LAM1)
    kapC = r.uniform(0.8, 1.3, nc)
    goneC = (cw > 0) & (lamC < tau[ckc])
    gone = goneC[cid]

    # 留在字身上的：整体平移到字此刻的位置；快要掉的块沿粗块边界先裂开
    pos = np.array([c["pos"] for c in flying])
    spr = np.array([c["sprite"][:2] for c in flying])
    shift_k = np.floor(pos) - spr
    near = np.clip(1 - (lamC[cid] - tau[kp]) / 0.06, 0, 1)
    crack = 1 - near * (1 - smoothstep(0.15, 1.0, cbd))
    sel = ~gone & (tau[kp] < LAM1)
    vals = np.stack([su[py, px], sc_[py, px]], 1)[sel] * (op_at(tau)[kp[sel]] * crack[sel])[:, None]
    ink.put((px[sel] + shift_k[kp[sel], 0]).astype(np.int64),
            (py[sel] + shift_k[kp[sel], 1]).astype(np.int64), vals)
    bodies = sum(1 for k, c in enumerate(flying) if c["tau"] < LAM1 and 0 <= c["pos"][0] < W and 0 <= c["pos"][1] < H)

    # 离开的粗块：离开时字在哪、转了多少、多大 → 起点；再沿风场漂到此刻
    cells = np.nonzero(goneC)[0]
    st = np.zeros((len(cells), 2))
    td = np.zeros(len(cells))
    for j, cell in enumerate(cells):
        c = flying[ck[cell]]
        gx, gy, fx, fy, rot_r, sc_r = c["sprite"]
        local = np.array([ccx[cell] - gx - fx, ccy[cell] - gy - fy])
        a = np.deg2rad(rot_at(c, lamC[cell]) - rot_r)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        td[j] = c["t0"] + flight_time(lamC[cell])
        st[j] = path_at(c, td[j]) + R @ local * (sc_at(lamC[cell]) / sc_r)
    end, hist = wind.advect(st, td, 0.0, speed=kapC[cells], steps=40, keep=10)
    dist = np.linalg.norm(end - st, axis=1)
    tang = end - hist[-2]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6
    Dc = np.zeros((nc, 2)); Dc[cells] = end - np.stack([ccx[cells], ccy[cells]], 1)
    Lc = np.zeros(nc); Lc[cells] = dist
    Tc = np.zeros((nc, 2)); Tc[cells] = tang
    # 细块：留不留（按距离指数衰减）和各自的散开量
    f_c = np.zeros(nf, np.int64)
    f_c[fid] = cid
    fL = Lc[f_c]
    keep_f = r.random(nf) < np.exp(-fL / 480.0)
    spread = np.clip(fL / 520.0, 0, 1) ** 1.2
    ft_, fn_ = Tc[f_c], np.stack([-Tc[f_c][:, 1], Tc[f_c][:, 0]], 1)
    f_off = ft_ * (r.normal(0, 1, nf) * 34 * spread)[:, None] + fn_ * (r.normal(0, 1, nf) * 8 * spread)[:, None]

    pg = gone & keep_f[fid]
    cc, ff = cid[pg], fid[pg]
    sp2 = spread[ff]
    off = np.stack([px[pg] + 0.5 - ccx[cc], py[pg] + 0.5 - ccy[cc]], 1)
    npg = int(pg.sum())
    jit = Tc[cc] * (r.normal(0, 1, npg) * 5 * sp2 ** 2)[:, None] + \
        np.stack([-Tc[cc][:, 1], Tc[cc][:, 0]], 1) * (r.normal(0, 1, npg) * 1.5 * sp2 ** 2)[:, None]
    disp = Dc[cc] + f_off[ff] + off * (0.8 * sp2 ** 2)[:, None] + jit
    v = np.stack([su[py, px], sc_[py, px]], 1)[pg]
    thin = np.clip(Lc[cc] / 1600, 0, 0.5)[:, None]
    v = np.clip((v - thin) / (1 - thin), 0, 1)
    # 刚离开时沿粗块边界有一道细缝；细块散开后沿细块边界碎
    edge = smoothstep(0.1, 0.9, cbd[pg]) * (1 - sp2) + smoothstep(0.1, 0.8, fbd[pg]) * sp2
    fade = op_at(lamC[cc]) * np.exp(-Lc[cc] / 1100) * edge
    d = np.round(disp).astype(np.int64)
    ink.put(px[pg] + d[:, 0], py[pg] + d[:, 1], v * fade[:, None])
    # 碎块身后的墨粉
    uu = np.bincount(cid, weights=su[py, px], minlength=nc)
    voice = (uu < 0.5 * cw).astype(np.int64)
    emit_dust(ink, st, hist, end, cw[cells] * np.exp(-dist / 900), voice[cells], op_at(lamC[cells]), r, 0.018)
    print(f"  飞字：字身可辨 {bodies} 个；{len(cells)} 个粗块飞散，留下细块 {int(keep_f[np.unique(fid[gone])].sum())}  "
          f"{time.time() - t0:.1f}s")


def main():
    t_all = time.time()
    out_png = ROOT / "04-螺.png"
    chars = layout()
    disperse(chars)
    n_state = [sum(1 for c in chars if c["ch"] and lo < c["D"] <= hi) for lo, hi in
               ((-1, D_ERODE), (D_ERODE, D_LOOSE), (D_LOOSE, D_FLY), (D_FLY, 2))]
    print(f"  完好 {n_state[0]}，蚀 {n_state[1]}，松动 {n_state[2]}，脱离 {n_state[3]}")
    wind = Wind(SEED)
    flying = fly(chars, wind)

    A = svg_rgb(build_inplace(chars), name="04_inplace")
    sp_body, sw, sh = build_sprites(flying)
    S = svg_rgb(sp_body, w=sw, h=sh, name="04_sprites")

    ink = Ink()
    mu, mc = inplace_layer(chars, A, wind, ink)
    flying_layer(flying, S, wind, ink)
    streak_dust(chars, flying, wind, ink)
    fly_u, fly_c = ink.acc[0].reshape(H, W), ink.acc[1].reshape(H, W)
    dust = ink.dust_cover()
    cov_u = np.maximum(mu, fly_u)
    cov_c = np.maximum(mc, fly_c)

    # ---- 纸、光
    pap = paper(H, W, PAPER, seed=SEED + 70, mottle=0.032, fiber=0.5, grain=0.010)
    lum = pap.mean(-1)
    dev = lum / gaussian(lum, 12) - 1
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1.0 + 0.04 * np.exp(-(((xx - 420) / 1900) ** 2 + ((yy - 380) / 2300) ** 2))
    light = light * vignette(H, W, strength=0.11, power=2.6, cx=0.46, cy=0.46)
    img = pap.copy()

    # ---- 墨：浓淡随一层很缓的低频场（不是逐字随机），纸纹凸起处略淡，边上极轻地洇开
    dens = 0.90 + 0.07 * smoothstep(-0.6, 0.6, fbm(H, W, SEED + 71, octaves=2, base=700))
    absorb = np.clip(1 - 2.2 * np.clip(dev, 0, None), 0.7, 1)
    tint = (pap / np.array(PAPER, np.float32)[None, None, :] * 255.0) ** 0.5

    def lay(img, cov, col):
        cov = np.clip(cov, 0, 1)
        cov = np.maximum(cov, 0.10 * gaussian(cov, 1.1))
        a = (cov * dens * absorb)[..., None]
        return img * (1 - a) + (col[None, None, :] * tint) * a

    img = lay(img, cov_u, INK_U)
    img = lay(img, cov_c, INK_C)
    img = lay(img, dust[0], INK_U)
    img = lay(img, dust[1], INK_C)

    # ---- 铅笔笔记：石墨只落在纸纹凸起处，所以有颗粒
    note = A[..., 2]
    tooth = unoise(H, W, SEED + 72, 1.4, 2)
    graph = note * (0.62 + 0.38 * smoothstep(0.15, 0.75, tooth)) * (0.9 + 0.6 * np.clip(dev, -0.1, 0.1))
    img = img * (1 - graph[..., None] * 0.92) + PENCIL[None, None, :] * (graph[..., None] * 0.92)

    img = img * light[..., None]
    save(np.clip(img, 0, 1), out_png, seed=SEED)
    print(f"  共 {time.time() - t_all:.1f}s")


def debug_dust():
    """只算粉尘（不排字），快速看风场的样子。"""
    chars = layout()
    disperse(chars)
    wind = Wind(SEED)
    flying = fly(chars, wind)
    ink = Ink()
    streak_dust(chars, flying, wind, ink)
    d = ink.dust_cover()
    img = np.full((H, W), 1.0, np.float32) * (1 - 0.9 * np.clip(d[0] + d[1], 0, 1))
    for c in chars:
        if c["ch"] and c["D"] <= D_FLY:
            cv2.circle(img, (int(c["x"]), int(c["y"])), int(c["s"] * 0.3), 0.55, -1)
    for c in flying:
        if c["tau"] < LAM1:
            cv2.circle(img, (int(c["pos"][0]), int(c["pos"][1])), int(c["s"] * 0.3), 0.3, -1)
    for p_, sig, gam in wind.vort:
        q = p_ + 0 * wind.DRIFT
        cv2.circle(img, (int(q[0]), int(q[1])), int(sig), 0.0 if gam > 0 else 0.6, 2)
    Image.fromarray((cv2.resize(img, (900, 1200), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)).save(WORK / "04_debug_dust.png")


if __name__ == "__main__":
    if "--dust" in sys.argv:
        debug_dust()
    else:
        main()
