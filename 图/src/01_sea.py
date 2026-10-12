"""图 1「海」：夜海和月亮，海面每一行都是人类写海的一句话。

    python -X utf8 01_sea.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import *          # noqa
from texts_sea import SEA, FOREGROUND, MOON, STARS, MILKY

SEED = 7
rng = np.random.default_rng(SEED)

YH = 1290.0                 # 地平线
CX = W / 2                  # 视线中心（月亮的 x）
K = 7400.0                  # 相机高度 × 焦距：y = YH + K / Z
F = K                       # 焦距（相机高度取 1）
RATIO = 1.03                # 相邻两行的深度比
Z_NEAR, Z_FAR = 3.72, 80.0
S0 = 168.0                  # 字号 = S0 / Z

# 海面高度场：(振幅, 波长, 方向角, 相位)。方向角 0 = 波峰平行于地平线
WAVES = [
    (0.034, 3.2, 1.33, 0.0),
    (0.020, 1.7, -1.40, 1.3),
    (0.011, 0.85, 1.46, 2.1),
    (0.006, 0.42, -1.50, 0.4),
    (0.004, 0.27, 1.53, 4.0),
]


def damp(Z):
    return np.minimum(1.0, 5.5 / Z)


def height(X, Z):
    h = 0.0
    for a, lam, ang, ph in WAVES:
        k = 2 * np.pi / lam
        h = h + a * np.sin(k * (np.sin(ang) * X + np.cos(ang) * Z) + ph)
    return h * damp(Z)


def slope_z(X, Z):
    """∂h/∂Z（只取波动项，忽略衰减项的导数）：朝向观者的坡。"""
    s = 0.0
    for a, lam, ang, ph in WAVES:
        k = 2 * np.pi / lam
        s = s + a * k * np.cos(ang) * np.cos(k * (np.sin(ang) * X + np.cos(ang) * Z) + ph)
    return s * damp(Z)


def slope_x(X, Z):
    s = 0.0
    for a, lam, ang, ph in WAVES:
        k = 2 * np.pi / lam
        s = s + a * k * np.sin(ang) * np.cos(k * (np.sin(ang) * X + np.cos(ang) * Z) + ph)
    return s * damp(Z)


XS = np.arange(-160, W + 170, 10, dtype=np.float64)


def line_ys(Z):
    X = (XS - CX) * Z / F
    return YH + (K - F * height(X, Z)) / Z


def separate(ys_list, sizes, gap=1.16):
    """从最近的一行往远处推：每一行的基线至少比下面那行高 gap × 下面那行的字号，避免两行字叠在一起。"""
    from scipy.ndimage import gaussian_filter1d
    out = [ys_list[0]]
    for i in range(1, len(ys_list)):
        lim = out[-1] - gap * sizes[i - 1]
        y = np.minimum(ys_list[i], lim)
        y = gaussian_filter1d(y, 2.0)
        y = np.minimum(y, lim)
        out.append(y)
    return out


def path_d(ys):
    return "M " + " L ".join(f"{x:.1f} {y:.2f}" for x, y in zip(XS, ys))


# 洗过的句子牌堆：每一轮都是 SEA 的一个随机排列，这样相邻几行不会撞上同一句
DECK = [q for _ in range(40) for q in (lambda p: [SEA[j] for j in p])(rng.permutation(len(SEA)))]


def build_sea():
    zs = []
    z = Z_NEAR
    while z < Z_FAR:
        zs.append(z)
        z *= RATIO
    defs, body = [], []
    k = 0
    n = len(zs)
    sizes = [S0 / Z for Z in zs]
    ys_all = separate([line_ys(Z) for Z in zs], sizes)
    for i, Z in enumerate(zs):
        size = sizes[i]
        pid = f"L{i}"
        defs.append(f'<path id="{pid}" d="{path_d(ys_all[i])}"/>')
        if i < len(FOREGROUND):                         # 最近的几行：指定句子
            base = FOREGROUND[i]
            items, w, j = [], 0.0, 0
            while w < W + 400:
                lang, t = base[j % len(base)]
                items.append((lang, t))
                w += est_width(lang, t, size) + est_width("lat", "  ·  ", size)
                j += 1
        else:
            items, k = fill_line(DECK, k, W + 400, size)
        off = rng.uniform(0, 2.5) * size
        body.append(
            f'<text font-size="{size:.3f}" fill="#fff" xml:space="preserve">'
            f'<textPath href="#{pid}" startOffset="{off:.1f}">{tspans(items)}</textPath></text>')
    print(f"  海面 {n} 行，最近一行字号 {S0 / zs[0]:.1f}px，最远 {S0 / zs[-1]:.2f}px")
    return "\n".join(defs), "\n".join(body)


MOON_C = (CX, 640.0)
MOON_R = 168.0


def build_moon():
    cx, cy = MOON_C
    size = 7.4
    lead = 8.6
    body = [f'<clipPath id="disc"><circle cx="{cx}" cy="{cy}" r="{MOON_R}"/></clipPath>']
    rows = []
    y = cy - MOON_R + size * 0.9
    k = 0
    while y < cy + MOON_R + size:
        k = int(rng.integers(0, len(MOON)))
        items, k = fill_line(MOON, k, 2 * MOON_R + 200, size, sep="　")
        rows.append(f'<text x="{cx - MOON_R - rng.uniform(5, 160):.1f}" y="{y:.1f}" font-size="{size}" fill="#fff" '
                    f'xml:space="preserve">{tspans(items, sep="　", sep_lang="zh")}</text>')
        y += lead
    return body[0], f'<g clip-path="url(#disc)">{"".join(rows)}</g>'


MW_A, MW_B = np.array([-140.0, 1060.0]), np.array([1620.0, -120.0])
MW_HALF = 170.0


def build_milky():
    """一条斜跨天空的天河：许多平行的细行，写的是关于银河的句子。"""
    d = MW_B - MW_A
    L = float(np.hypot(*d))
    u = d / L
    nrm = np.array([-u[1], u[0]])
    defs, body = [], []
    k = 0
    off = -MW_HALF * 1.4
    i = 0
    while off < MW_HALF * 1.4:
        size = float(rng.uniform(3.6, 5.4))
        a = MW_A + nrm * off
        b = MW_B + nrm * off
        # 稍微弯一点
        mid = (a + b) / 2 + nrm * 40
        defs.append(f'<path id="M{i}" d="M {a[0]:.1f} {a[1]:.1f} Q {mid[0]:.1f} {mid[1]:.1f} {b[0]:.1f} {b[1]:.1f}"/>')
        items, k = fill_line(MILKY, k + int(rng.integers(0, 4)), L + 100, size, sep="  ")
        body.append(f'<text font-size="{size:.2f}" fill="#fff" xml:space="preserve"><textPath href="#M{i}" '
                    f'startOffset="{rng.uniform(0, 30):.1f}">{tspans(items, sep="  ")}</textPath></text>')
        off += size * rng.uniform(1.05, 1.5)
        i += 1
    return "\n".join(defs), "\n".join(body)


def milky_profile(xx, yy):
    d = MW_B - MW_A
    L = float(np.hypot(*d))
    u = d / L
    nrm = np.array([-u[1], u[0]])
    t = ((xx - MW_A[0]) * u[0] + (yy - MW_A[1]) * u[1]) / L
    # 弯曲的中线：二次贝塞尔的偏移约为 4t(1-t)·40/2
    o = (xx - MW_A[0]) * nrm[0] + (yy - MW_A[1]) * nrm[1] - 40 * 2 * t * (1 - t)
    across = np.exp(-(o / MW_HALF) ** 2)
    along = smoothstep(-0.05, 0.25, t) * (1 - smoothstep(0.75, 1.05, t))
    return across * along, o


def build_stars():
    out = []
    n = 0
    tries = 0
    pts = []
    while n < 170 and tries < 5000:
        tries += 1
        x = rng.uniform(40, W - 40)
        y = rng.uniform(40, YH - 120)
        # 越靠近地平线越稀，月亮周围不放
        if rng.random() < (y / YH) ** 1.6:
            continue
        if np.hypot(x - MOON_C[0], y - MOON_C[1]) < MOON_R * 2.1:
            continue
        if any(np.hypot(x - px, y - py) < 70 for px, py in pts):
            continue
        pts.append((x, y))
        lang, t = STARS[int(rng.integers(0, len(STARS)))]
        size = float(rng.choice([5.5, 6.5, 7.5, 9, 11, 14], p=[0.3, 0.25, 0.2, 0.13, 0.08, 0.04]))
        op = float(np.clip(rng.uniform(0.35, 1.0) * (size / 9) ** 0.5, 0.2, 1))
        out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="{FONT[lang]}" '
                   f'fill="#fff" fill-opacity="{op:.2f}" text-anchor="middle">{esc(t)}</text>')
        n += 1
    return "\n".join(out)


def main():
    out_png = ROOT / "01-海.png"
    sea_defs, sea_body = build_sea()
    sea = svg_mask(sea_body, defs=sea_defs, name="01_sea_text")
    moon_clip, moon_body = build_moon()
    moon = svg_mask(moon_body, defs=moon_clip, name="01_moon_text")
    stars = svg_mask(build_stars(), name="01_stars")
    mw_defs, mw_body = build_milky()
    milky = svg_mask(mw_body, defs=mw_defs, name="01_milky")

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    below = yy > YH

    # ---- 天空
    t = np.clip(yy / YH, 0, 1)
    sky = lerp(rgbf("#03060c"), rgbf("#18233a"), (t ** 2.2)[..., None])
    dm = np.hypot(xx - MOON_C[0], yy - MOON_C[1])
    halo = 0.10 * np.exp(-((dm - MOON_R) / 260.0).clip(0) ** 1.0) + 0.05 * np.exp(-(dm / 900.0) ** 2)
    sky = sky + halo[..., None] * rgbf("#9fb2d6")

    # ---- 天河：带状亮度 × 云气噪声，中间一道暗尘带
    prof, o = milky_profile(xx, yy)
    cloud = smoothstep(-0.5, 0.75, fbm(H, W, SEED + 21, octaves=6, base=260))
    lane = 1 - 0.75 * np.exp(-((o + 18 + 30 * fbm(H, W, SEED + 22, octaves=3, base=300)) / 26.0) ** 2)
    mw = prof * cloud * lane * (yy < YH - 60)
    sky = sky + (0.035 * mw)[..., None] * rgbf("#b9c6e6")
    sky = sky * (1 - (milky * mw * 0.0)[..., None]) + (milky * mw * 0.42)[..., None] * rgbf("#d6def0")

    # ---- 海面底色（字与字之间的水）
    zpix = np.where(below, K / np.maximum(yy - YH, 0.5), 1e3)
    depth = np.clip(np.log(zpix / Z_NEAR) / np.log(Z_FAR / Z_NEAR), 0, 1)   # 0 近 1 远
    water = lerp(rgbf("#050a14"), rgbf("#0f1829"), (depth ** 2.0)[..., None])
    img = np.where(below[..., None], water, sky)

    # ---- 月光在海面上的路：按法线算反射，看反射方向离月亮多近（闪烁），再乘一个柱状包络
    X = (xx - CX) * zpix / F
    rough = 1.0 + 2.6 * (1 - depth) ** 1.5          # 近处的水更「碎」，好让月光一路铺到脚下
    sx, sz = slope_x(X, zpix) * rough, slope_z(X, zpix) * rough
    d = np.stack([(xx - CX) / F, -(yy - YH) / F, np.ones_like(xx)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    n = np.stack([-sx, np.ones_like(sx), -sz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    r = d - 2 * (d * n).sum(-1, keepdims=True) * n
    th = (YH - MOON_C[1]) / F
    m = np.array([0.0, np.sin(th), np.cos(th)], np.float32)
    cosang = (r * m).sum(-1)
    glint = np.exp(-(1 - cosang) / 0.0016) * below
    width = 30 + 0.36 * np.clip(yy - YH, 0, None)
    env = np.exp(-((xx - CX) / width) ** 2)
    sparkle = 0.6 + 0.4 * fbm(H, W, SEED + 5, octaves=4, base=20)
    moonlight = (0.22 * env + 2.6 * glint * sparkle * (0.35 + 0.65 * env)) * below
    path_glow = (0.045 * np.exp(-((xx - CX) / (width * 0.8)) ** 2)
                 + 0.10 * np.exp(-((xx - CX) / 26) ** 2) * np.exp(-((yy - YH) / 120) ** 2)) * below
    img = img + path_glow[..., None] * rgbf("#c9b48a")
    near_glow = env * below * 0.10

    # ---- 海面上的字
    base_col = lerp(rgbf("#4f607d"), rgbf("#73839f"), (depth ** 1.8)[..., None])
    gold = rgbf("#ffcd7e")
    text_col = base_col * (0.85 + near_glow[..., None]) + gold * moonlight[..., None]
    img = img * (1 - sea[..., None] * below[..., None]) + text_col * (sea * below)[..., None]
    # 远处的行糊成雾：地平线上一道薄亮
    haze = np.exp(-((yy - YH - 6) / 14.0) ** 2) * 0.10
    img = img + haze[..., None] * rgbf("#8a9bbb")

    # ---- 月亮：字组成的月面，有月海（暗斑）和边缘变暗
    r = dm / MOON_R
    inside = np.clip((1 - r) * MOON_R, 0, 1)
    maria = fbm(H, W, SEED + 9, octaves=5, base=90)
    maria = smoothstep(0.05, 0.45, maria)
    limb = np.sqrt(np.clip(1 - r * r, 0, 1)) ** 0.35
    moon_bright = (1.9 - 0.75 * maria) * limb
    disc_glow = inside * (0.30 - 0.10 * maria) * limb
    moon_col = rgbf("#fff2d6")
    img = img + (disc_glow[..., None] * rgbf("#d9d2bd"))
    img = img * (1 - (moon * inside)[..., None]) + (moon_col * (moon * inside * moon_bright)[..., None])

    # ---- 星
    img = img + stars[..., None] * rgbf("#dfe6f5") * (yy < YH)[..., None] * 0.95

    # ---- 光晕、暗角、纸
    img = bloom(img, sigmas=(3, 12, 40, 120), weights=(0.30, 0.22, 0.14, 0.08), threshold=0.30)
    img = img * vignette(H, W, strength=0.30, cy=0.45)[..., None]
    grain = paper(H, W, (1.0, 1.0, 1.0), seed=SEED + 3, mottle=0.05, fiber=0.6, grain=0.02)
    img = img * (0.94 + 0.06 * grain)
    img = img / (1 + 0.18 * img)              # 轻微压高光
    img = img * 1.12
    save(np.clip(img, 0, 1), out_png, seed=SEED)


if __name__ == "__main__":
    main()
