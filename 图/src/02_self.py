"""图 2「我」：一页写满人类写「自己」的句子的手稿，一个大毛笔「我」从这页字里浓出来。

做法：整页 10–11 px 的淡字由 Chrome 排（行尾用 SVG 里的一段 JS 按词截断）。「我」里另排一层更密更重的字
（同样的句子，Noto Serif SC 900，约 10.3 px，行距 10.7 px，描 0.4 / 0.9 px 边的两层同排版），只在「我」里出现；
外面的淡字在笔画外形以内让位，所以飞白处露的是纸。「我」用华文行楷在 PIL 里画蒙版，加低频扭曲、纤维毛边，
再沿笔势方向场（结构张量）做线积分卷积得到顺着笔势的丝缕，在手工指定的收笔处做飞白。浓淡来自笔画内部：
笔画粗处（最近骨架点的半宽）和手工标的起笔 / 顿笔 / 转折 / 交叉处用更重的那层字、洇墨更多，中段和收笔轻。
「我」里垫底的淡墨只有 WASH=0.08（只让字缝不发白），黑主要靠字本身：「我」里平均亮度约 0.47，页面 0.785。
纸、掠射光、暗角、朱红「存疑」竖排长方印在 numpy 里合成。

    python -X utf8 02_self.py     # 文字蒙版有缓存（work/02_text_*.npy、02_dense*_*.npy），改了排版才会重跑 Chrome
"""
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numba import njit, prange
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent))
from common import *          # noqa
from texts_self import SELF

SEED = 22
rng = np.random.default_rng(SEED)

PAPER = "#efe6d3"
INK = "#1c1712"
X0, X1, Y0, Y1 = 140, 2260, 140, 3060      # 版心
LEAD = 14.0
FONTS = "C:/Windows/Fonts/"

# 「我」：宽 = 版心的 75%，中心略高于页面中心
GLYPH_W = 0.75 * (X1 - X0)
GLYPH_C = (1200.0, 1430.0)
WASH, HALO = 0.08, 0.55           # 「我」里垫的淡墨（几乎不用）、密字周围洇墨的强度
OUT_A = 0.58                      # 外面淡字的墨


# ---------------------------------------------------------------- 整页小字

JS_CUT = r"""<script><![CDATA[
(function(){
function run(){
  var lines = document.querySelectorAll('text.ln');
  for (var li = 0; li < lines.length; li++) {
    var t = lines[li];
    var lim = parseFloat(t.getAttribute('data-r'));
    var spans = t.querySelectorAll('tspan');
    var cutAt = -1;
    for (var i = 0; i < spans.length; i++) {
      var s = spans[i];
      var n = s.getNumberOfChars();
      if (!n) continue;
      var e0 = s.getExtentOfChar(0), e1 = s.getExtentOfChar(n - 1);
      var r = Math.max(e0.x + e0.width, e1.x + e1.width);
      if (r <= lim) continue;
      var lang = s.getAttribute('data-l');
      cutAt = i;
      if (lang === 'zh' || lang === 'ja' || lang === 'lat') {
        var k = 0;
        while (k < n) { var e = s.getExtentOfChar(k); if (e.x + e.width > lim) break; k++; }
        var txt = s.textContent, keep = k;
        if (lang === 'lat' && txt.charAt(k) !== ' ') {
          var sp = txt.lastIndexOf(' ', k - 1);
          keep = sp > 0 ? sp : 0;
        }
        txt = txt.slice(0, keep).replace(/[\s,;:]+$/, '');
        if (txt.length > 1) { s.textContent = txt; cutAt = i + 1; }
      }
      break;
    }
    if (cutAt >= 0) {
      for (var j = spans.length - 1; j >= cutAt; j--) spans[j].remove();
      var rem = t.querySelectorAll('tspan');
      var last = rem[rem.length - 1];
      if (last && last.getAttribute('data-l') === 'sep') last.remove();
    }
  }
  document.documentElement.setAttribute('data-done', '1');
}
if (document.fonts && document.fonts.ready) document.fonts.ready.then(run); else run();
})();
]]></script>"""


def line_spans(items, sep="  ·  "):
    """[(lang, text)] → tspans；每句的基线有零点几像素的起伏（像手写）。"""
    out, prev = [], 0.0
    for k, (lang, t) in enumerate(items):
        if k:
            out.append(f'<tspan data-l="sep" font-family="{FONT["lat"]}">{esc(sep)}</tspan>')
        d = float(rng.normal(0, 0.28))
        out.append(f'<tspan data-l="{lang}" font-family="{FONT[lang]}" dy="{d - prev:.2f}">{esc(t)}</tspan>')
        prev = d
    return "".join(out)


_MEASURE_FONTS = {"zh": ("NotoSerifSC-VF.ttf", 0), "ja": ("NotoSerifSC-VF.ttf", 0), "lat": ("cambria.ttc", 0),
                  "he": ("times.ttf", 0), "hi": ("Nirmala.ttc", 0)}
_MEASURE_HEAVY = {"zh": "NotoSerifSC-VF.ttf", "ja": "NotoSerifSC-VF.ttf", "lat": "NotoSerifSC-VF.ttf",
                  "he": "timesbd.ttf", "hi": "Nirmala.ttc"}
_ADV_CACHE = {}


def adv(lang, text, size, heavy=False):
    """用和 Chrome 相同的字体量一句话的前进宽度（只用来排「别让同一句在相邻行对齐」，截断交给 Chrome）。"""
    key = (lang, text, heavy)
    if key not in _ADV_CACHE:
        if heavy:
            f = _MEASURE_HEAVY[lang]
            ft = ImageFont.truetype(FONTS + f, 100, layout_engine=ImageFont.Layout.RAQM)
            if f.startswith("NotoSerifSC"):
                ft.set_variation_by_axes([900])
        else:
            f, idx = _MEASURE_FONTS[lang]
            ft = ImageFont.truetype(FONTS + f, 100, index=idx, layout_engine=ImageFont.Layout.RAQM)
        _ADV_CACHE[key] = ft.getlength(text) / 100.0
    return _ADV_CACHE[key] * size


def plan_line(k, x0, length, size, sep="  ·  ", heavy=False, over=1.3):
    """从第 k 句起排一行：返回 [(句序号, 起点 x)]，排到 length 的 over 倍为止（多出来的由 JS 截掉）。"""
    out, x, j = [], x0, k
    sw = adv("lat", sep, size, heavy)
    while x < x0 + length * over:
        lang, t = SELF[j % len(SELF)]
        out.append((j % len(SELF), x))
        x += adv(lang, t, size, heavy) + sw
        j += 1
    return out


def clashes(plan, prev, tol):
    """plan 里有没有哪一句和 prev 里同一句的起点相差不到 tol（也就是上下对齐成一列）。"""
    pos = {}
    for i, x in prev:
        pos.setdefault(i, []).append(x)
    for i, x in plan:
        for xp in pos.get(i, ()):
            if abs(x - xp) < tol:
                return True
    return False


def build_page():
    rows = []
    y = Y0 + 9.5
    hist = []          # 前几行的排法
    n, retries = 0, 0
    while y < Y1 - 2:
        size = float(rng.uniform(10.0, 11.0))
        x0 = X0 + float(rng.uniform(-2.0, 5.0))
        lim = X1 - float(rng.uniform(0.0, 9.0))
        for attempt in range(400):
            k = int(rng.integers(0, len(SELF)))
            plan = [(i, x) for i, x in plan_line(k, x0, lim - x0, size) if x < lim]
            # 上一行同一句的起点至少错开 260 px，再往上几行依次放宽
            if not any(clashes(plan, hp, tol) for hp, tol in zip(reversed(hist), (260, 160, 110, 80))):
                break
            retries += 1
        hist = (hist + [plan])[-4:]
        items, _ = fill_line(SELF, k, (lim - x0) * 1.3, size)
        yy = y + float(rng.normal(0, 0.35))
        rot = float(rng.normal(0, 0.02))
        rows.append(
            f'<text class="ln" x="{x0:.2f}" y="{yy:.2f}" data-r="{lim:.1f}" font-size="{size:.2f}" fill="#fff" '
            f'xml:space="preserve" transform="rotate({rot:.4f} {x0:.1f} {yy:.1f})">{line_spans(items)}</text>')
        y += LEAD * float(rng.uniform(0.985, 1.015))
        n += 1
    print(f"  正文 {n} 行（为避开上下对齐重抽 {retries} 次）")
    return "\n".join(rows) + JS_CUT


def text_masks():
    """整页淡字的蒙版；按 SVG 内容的哈希缓存。"""
    body = build_page()
    key = hashlib.md5(body.encode("utf-8")).hexdigest()[:10]
    p0 = WORK / f"02_text_{key}.npy"
    if p0.exists():
        print("  文字蒙版：用缓存", key)
        return np.load(p0)
    t0 = svg_mask(body, name="02_text")
    np.save(p0, t0.astype(np.float32))
    return t0


# ---------------------------------------------------------------- 「我」里的密字

# 「我」里单独排一层更密更重的字：同样的 SELF，Noto Serif SC 900，约 10 px，行距约 11 px，
# 行位置和外面的淡字无关；只在「我」的蒙版里出现。
DX0, DX1, DY0, DY1 = 350, 2050, 440, 2440
D_LEAD = 10.7
D_SIZE = (10.0, 10.6)
D_SEP = " · "
FONT_D = {"zh": "'Noto Serif SC'", "ja": "'Noto Serif SC'", "lat": "'Noto Serif SC', Cambria, serif",
          "he": "'Times New Roman'", "hi": "'Nirmala UI'"}


def build_dense():
    r = np.random.default_rng(SEED + 200)
    rows, hist = [], []
    y = DY0 + 8.0
    n, retries = 0, 0
    while y < DY1:
        size = float(r.uniform(*D_SIZE))
        x0 = DX0 - float(r.uniform(0, 40))
        length = DX1 - x0
        for attempt in range(400):
            k = int(r.integers(0, len(SELF)))
            plan = plan_line(k, x0, length, size, D_SEP, heavy=True, over=1.0)
            if not any(clashes(plan, hp, tol) for hp, tol in zip(reversed(hist), (220, 140, 100, 70))):
                break
            retries += 1
        hist = (hist + [plan])[-4:]
        spans, prev = [], 0.0
        for j, (i, _) in enumerate(plan):
            lang, t = SELF[i]
            if j:
                spans.append(f'<tspan font-family="{FONT_D["zh"]}">{esc(D_SEP)}</tspan>')
            d = float(r.normal(0, 0.22))
            spans.append(f'<tspan font-family="{FONT_D[lang]}" dy="{d - prev:.2f}">{esc(t)}</tspan>')
            prev = d
        yy = y + float(r.normal(0, 0.3))
        rot = float(r.normal(0, 0.02))
        rows.append(f'<text x="{x0:.2f}" y="{yy:.2f}" font-size="{size:.2f}" font-weight="900" fill="#fff" '
                    f'xml:space="preserve" transform="rotate({rot:.4f} {x0:.1f} {yy:.1f})">{"".join(spans)}</text>')
        y += D_LEAD * float(r.uniform(0.985, 1.015))
        n += 1
    print(f"  「我」里的密字 {n} 行（重抽 {retries} 次）")
    return f'<g transform="translate({-DX0} {-DY0})">' + "\n".join(rows) + "</g>"


def dense_masks():
    """「我」里的密字：(900 描 0.4 px 边, 900 描 0.9 px 边)，同一排版，按墨的足否在两层间过渡。按内容哈希缓存。"""
    body = build_dense()
    base_css = "text{stroke:#fff;stroke-width:0.4px;stroke-linejoin:round}"
    heavy_css = "text{stroke:#fff;stroke-width:0.9px;stroke-linejoin:round}"
    key = hashlib.md5((body + base_css + heavy_css).encode("utf-8")).hexdigest()[:10]
    p0, p1 = WORK / f"02_dense_{key}.npy", WORK / f"02_denseheavy_{key}.npy"
    if p0.exists() and p1.exists():
        print("  密字蒙版：用缓存", key)
        return np.load(p0), np.load(p1)
    out = []
    for name, css in (("02_dense", base_css), ("02_denseheavy", heavy_css)):
        sub = svg_mask(body, w=DX1 - DX0, h=DY1 - DY0, name=name, css=css)
        full = np.zeros((H, W), np.float32)
        full[DY0:DY1, DX0:DX1] = sub
        out.append(full)
    np.save(p0, out[0])
    np.save(p1, out[1])
    return out[0], out[1]


# ---------------------------------------------------------------- 「我」

def glyph_raw():
    """华文行楷「我」，按实际墨迹的外框缩放到 GLYPH_W 宽，放到 GLYPH_C。返回 [0,1] 覆盖率。"""
    ss = 2
    ft = ImageFont.truetype(FONTS + "STXINGKA.TTF", 1100 * ss)
    can = Image.new("L", (1800 * ss, 1800 * ss), 0)
    ImageDraw.Draw(can).text((200 * ss, 100 * ss), "我", font=ft, fill=255)
    a = np.asarray(can).astype(np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.02)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = GLYPH_W / a.shape[1]
    gw, gh = int(round(a.shape[1] * s)), int(round(a.shape[0] * s))
    g = cv2.resize(a, (gw, gh), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W), np.float32)
    x0, y0 = int(round(GLYPH_C[0] - gw / 2)), int(round(GLYPH_C[1] - gh / 2))
    out[y0:y0 + gh, x0:x0 + gw] = g
    print(f"  「我」外框 x {x0}–{x0 + gw}，y {y0}–{y0 + gh}")
    return out


def warp(img, amp, base, seed):
    dx = fbm(H, W, seed, octaves=3, base=base) * amp
    dy = fbm(H, W, seed + 1, octaves=3, base=base) * amp
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def direction_field(m, sigma_g=4.0, sigma_i=40.0):
    """结构张量 → 笔势切向 (tx, ty)，单位向量。"""
    g = gaussian(m, sigma_g)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = gaussian(gx * gx, sigma_i)
    jxy = gaussian(gx * gy, sigma_i)
    jyy = gaussian(gy * gy, sigma_i)
    th = 0.5 * np.arctan2(2 * jxy, jxx - jyy)       # 梯度方向（横穿笔画）
    return (-np.sin(th)).astype(np.float32), np.cos(th).astype(np.float32)


@njit(cache=True)
def _bilinear(a, x, y):
    h, w = a.shape
    if x < 0 or y < 0 or x > w - 1.001 or y > h - 1.001:
        return 0.0, False
    ix, iy = int(x), int(y)
    fx, fy = x - ix, y - iy
    v = (a[iy, ix] * (1 - fx) * (1 - fy) + a[iy, ix + 1] * fx * (1 - fy)
         + a[iy + 1, ix] * (1 - fx) * fy + a[iy + 1, ix + 1] * fx * fy)
    return v, True


@njit(parallel=True, cache=True)
def lic(src, tx, ty, region, L):
    """线积分卷积：沿 (tx, ty) 流线前后各走 L 步（1 px），三角权重平均 src。只算 region 内的像素。"""
    h, w = src.shape
    out = np.zeros_like(src)
    for y in prange(h):
        for x in range(w):
            if not region[y, x]:
                continue
            acc = src[y, x]
            nw = 1.0
            for sgn in (1.0, -1.0):
                px, py = x + 0.0, y + 0.0
                dx0, dy0 = tx[y, x] * sgn, ty[y, x] * sgn
                for s in range(L):
                    ix, iy = int(px + 0.5), int(py + 0.5)
                    if ix < 0 or iy < 0 or ix >= w or iy >= h:
                        break
                    dx, dy = tx[iy, ix], ty[iy, ix]
                    if dx * dx0 + dy * dy0 < 0:
                        dx, dy = -dx, -dy
                    px += dx
                    py += dy
                    dx0, dy0 = dx, dy
                    v, ok = _bilinear(src, px, py)
                    if not ok:
                        break
                    wt = 1.0 - s / L
                    acc += v * wt
                    nw += wt
            out[y, x] = acc / nw
    return out


# 收笔处的飞白：(x, y, 半径, 强度)。位置是看了「我」的字形后手工定的（笔势快、墨将尽的地方）。
DRY = [
    (1820, 2160, 190, 0.75),   # 斜钩下段
    (1940, 2330, 90, 0.80),    # 斜钩的尖
    (1898, 1890, 110, 0.85),   # 钩往上挑的出锋
    (1170, 1845, 140, 0.90),   # 右边那一撇的撇尾
    (1005, 2180, 110, 0.80),   # 竖的下端
    (455, 1390, 240, 0.70),    # 左边大弧的外沿
    (1420, 1460, 150, 0.55),   # 提的出锋
    (1700, 1620, 200, 0.40),   # 斜钩中段，轻微
    (1080, 560, 70, 0.45),     # 起笔那一撇的尖
]

# 甩出来的墨点：(x, y, 半径, 方向角°)
SPLATTER = [
    (1983, 1770, 6.5, -70), (1999, 1741, 3.5, -70), (1967, 1722, 2.5, -75),
    (1640, 1168, 7, 5), (1676, 1160, 4, 0),
    (2030, 2400, 6, 40), (2060, 2425, 3, 40),
]


def ink_mask():
    """「我」的墨：[0,1]，1 = 浓墨。含软边、毛边、飞白、墨点。"""
    t = time.time()
    g = glyph_raw()
    g = warp(g, 6.0, 260, SEED + 40)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    region = cv2.dilate((g > 0.02).astype(np.uint8), np.ones((121, 121), np.uint8)) > 0

    tx, ty = direction_field(g)

    # 刷毛的丝缕：细噪声 + 粗一点的噪声，沿笔势拉长
    n1 = gaussian(rng.standard_normal((H, W)).astype(np.float32), 1.5)
    n1 /= n1.std() + 1e-6
    n2 = gaussian(rng.standard_normal((H, W)).astype(np.float32), 6.0)
    n2 /= n2.std() + 1e-6
    noise = n1 * 0.5 + n2 * 1.0
    streak = lic(noise, tx, ty, region, 70)
    streak = (streak - streak[region].mean()) / (streak[region].std() + 1e-6)

    # 干湿：手工收笔区 + 笔画边缘
    dry = np.zeros((H, W), np.float32)
    for x, y, r, s in DRY:
        dry = np.maximum(dry, s * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (r * r)))
    inside = (g > 0.5).astype(np.uint8)
    dt = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    edge = np.exp(-dt / 5.0)
    dry = np.clip(dry + 0.35 * edge * (0.4 + 0.6 * dry), 0, 1)

    # 收笔处刷毛顺着笔势拖出去一点
    smear = lic(g, tx, ty, region, 45)
    base = np.maximum(g, np.clip(smear * 1.4 - 0.2, 0, 1) * smoothstep(0.4, 0.9, dry))

    # 软边 + 纤维毛边
    fib = fbm(H, W, SEED + 41, octaves=3, base=10)
    base = smoothstep(0.30, 0.70, gaussian(base, 2.0) + 0.10 * fib)

    # 纸面的细颗粒让干笔的丝缕断断续续
    tooth = lic(gaussian(rng.standard_normal((H, W)).astype(np.float32), 0.8), tx, ty, region, 9)
    tooth = (tooth - tooth[region].mean()) / (tooth[region].std() + 1e-6)
    thr = lerp(-3.4, 0.45, dry)
    keep = smoothstep(thr - 0.25, thr + 0.25, streak + 0.22 * tooth * smoothstep(0.1, 0.5, dry))
    density = 1.0 - 0.22 * dry
    m = base * keep * density

    # 墨点
    for x, y, r, ang in SPLATTER:
        a = np.deg2rad(ang)
        u = (xx - x) * np.cos(a) + (yy - y) * np.sin(a)
        v = -(xx - x) * np.sin(a) + (yy - y) * np.cos(a)
        d = np.sqrt((u / 1.35) ** 2 + v ** 2) / r
        dot = smoothstep(1.15, 0.85, d + 0.18 * fib)
        m = np.maximum(m, dot)
        base = np.maximum(base, dot)
    print(f"  「我」蒙版 {time.time() - t:.1f}s")
    # m：墨（含飞白）；base：笔画的外形（飞白之前），用来让外面的淡字在笔画里让位、飞白处露纸
    return np.clip(m, 0, 1).astype(np.float32), dry, g, base.astype(np.float32)


# 起笔、顿笔、转折、交叉处墨更足：(x, y, 半径, 强度)
HEAVY = [
    (1080, 610, 90, 0.70),     # 起笔那一撇
    (1010, 830, 90, 0.80),     # 横的起笔
    (1555, 1180, 110, 0.90),   # 横收笔的顿
    (870, 1240, 90, 0.80),     # 竖的起笔
    (650, 1880, 90, 0.70),     # 提的起笔
    (1560, 1420, 90, 0.80),    # 右撇的起笔
    (1730, 955, 100, 0.85),    # 点
    (500, 1470, 130, 0.60),    # 左边大弧的转折
    (1330, 1170, 110, 0.60),   # 横与斜钩相交
    (900, 1660, 90, 0.60),     # 提与竖相交
    (1180, 720, 100, 0.60),    # 斜钩的起笔
]


def heaviness(g, dry):
    """[0,1]：笔画哪里墨足、字重。粗处（行楷的起笔和顿笔本来就粗）+ 手工标的起笔 / 转折 / 交叉，收笔处的干减掉。"""
    inside = g > 0.5
    dt = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(inside)
    _, (iy, ix) = ndimage.distance_transform_edt(~sk, return_indices=True)
    inf = inside.astype(np.float32)
    half = gaussian(dt[iy, ix] * inf, 30.0) / (gaussian(inf, 30.0) + 1e-4)   # 最近骨架点的半宽 = 这里笔画多粗
    thick = smoothstep(np.percentile(half[inside], 25), np.percentile(half[inside], 92), half)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    spots = np.zeros((H, W), np.float32)
    for x, y, r, s in HEAVY:
        spots = np.maximum(spots, s * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (r * r)))
    hv = np.clip(0.5 * thick + spots, 0, 1)
    return (0.25 + 0.75 * hv) * (1 - 0.6 * dry)


# ---------------------------------------------------------------- 印

SEAL_C = (2152.0, 2924.0)
SEAL_W, SEAL_H = 108, 180


def seal():
    """白文长方印「存疑」，竖排：上「存」下「疑」。字是华文隶书。返回 (覆盖率, 颜色 RGB)。"""
    ss = 4
    SW, SH = SEAL_W * ss, SEAL_H * ss
    pad = 40 * ss
    can = Image.new("L", (SW + 2 * pad, SH + 2 * pad), 0)
    ImageDraw.Draw(can).rectangle([pad, pad, pad + SW, pad + SH], fill=255)
    ft = ImageFont.truetype(FONTS + "STLITI.TTF", int(SW * 1.1))
    inner = SW * 0.11
    gap = SW * 0.05
    cw, chh = SW - 2 * inner, (SH - 2 * inner - gap) / 2
    arr = np.asarray(can).copy()
    for ch, row in (("存", 0), ("疑", 1)):
        c = Image.new("L", (int(SW * 1.6), int(SW * 1.6)), 0)
        ImageDraw.Draw(c).text((0, 0), ch, font=ft, fill=255)
        a = np.asarray(c)
        ys, xs = np.nonzero(a > 10)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        tw, th = int(cw * 0.95), int(chh * 0.95)
        a = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)
        ox = int(pad + inner + (cw - tw) / 2)
        oy = int(pad + inner + row * (chh + gap) + (chh - th) / 2)
        arr[oy:oy + th, ox:ox + tw] = np.minimum(arr[oy:oy + th, ox:ox + tw], 255 - a)
    can = Image.fromarray(arr)
    a = np.asarray(can).astype(np.float32) / 255.0
    a = cv2.resize(a, (a.shape[1] // ss, a.shape[0] // ss), interpolation=cv2.INTER_AREA)
    h, w = a.shape
    # 残缺：外缘用噪声侵蚀；一角磨掉一块
    r = np.random.default_rng(SEED + 60)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n_big = cv2.resize(r.standard_normal((h // 12 + 2, w // 12 + 2)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    n_fine = gaussian(r.standard_normal((h, w)).astype(np.float32), 1.2) * 2.5
    edge_d = np.minimum.reduce([xx - 40, yy - 40, 40 + SEAL_W - xx, 40 + SEAL_H - yy])
    erode = smoothstep(12, 0, edge_d) * (0.6 * n_big + 0.45 * n_fine + 0.25)
    corner = np.exp(-(((xx - (40 + SEAL_W)) ** 2 + (yy - 40) ** 2) / (24.0 ** 2)))
    a = gaussian(a, 0.7)
    rough = gaussian(r.standard_normal((h, w)).astype(np.float32), 0.8) * 1.8
    a = smoothstep(0.35, 0.65, a + 0.05 * rough - np.clip(erode, 0, None) * 0.9 - corner * (0.6 + 0.4 * n_big))
    # 印泥不匀：纸纹里吃不进去的地方淡、斑驳
    blot = fbm(h, w, SEED + 61, octaves=5, base=40)
    speck = gaussian(r.standard_normal((h, w)).astype(np.float32), 0.9)
    paste = np.clip(0.88 + 0.12 * blot - 0.22 * smoothstep(0.9, 1.8, speck), 0, 1)
    voids = smoothstep(1.4, 2.2, speck + 0.6 * blot)
    grain = gaussian(r.standard_normal((h, w)).astype(np.float32), 0.6) * 2.2
    a = a * paste * (1 - 0.6 * voids) * (1 - 0.25 * smoothstep(1.2, 2.4, grain))
    col = lerp(rgbf("#b0301f"), rgbf("#c8452a"), smoothstep(-0.5, 0.6, blot)[..., None])
    # 手盖的，略歪
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 1.6, 1.0)
    a = cv2.warpAffine(a, rot, (w, h), flags=cv2.INTER_LINEAR)
    col = cv2.warpAffine(col, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out_a = np.zeros((H, W), np.float32)
    out_c = np.zeros((H, W, 3), np.float32)
    x0, y0 = int(SEAL_C[0] - w / 2), int(SEAL_C[1] - h / 2)
    out_a[y0:y0 + h, x0:x0 + w] = a
    out_c[y0:y0 + h, x0:x0 + w] = col
    return out_a, out_c


# ---------------------------------------------------------------- 合成

def main():
    t_all = time.time()
    out_png = ROOT / "02-我.png"
    t0 = text_masks()
    d0, d1 = dense_masks()
    m, dry, g, outline = ink_mask()
    hv = heaviness(g, dry)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # ---- 纸
    pap = paper(H, W, tuple(rgbf(PAPER)), seed=SEED + 1, mottle=0.035, fiber=1.0, grain=0.012)

    # ---- 外面的淡字：笔画外形以内整个让位（飞白处露的是纸）；笔画外沿的字沾一点墨
    near = gaussian(m, 14.0)
    a_out = t0 * OUT_A * (1 + 0.8 * near) * (1 - smoothstep(0.0, 0.35, outline))
    # 印底下的字淡一些（印泥盖住大半），好让「存疑」两个白字读得出来
    sa, sc = seal()
    foot = gaussian(cv2.dilate((sa > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), 4.0)
    a_out = a_out * (1 - 0.6 * foot)
    img = pap * (1 - a_out[..., None]) + rgbf("#3a3027") * a_out[..., None]

    # ---- 「我」里的密字：墨足处（hv 大）字更重、墨更浓、洇得多；中段和收笔处轻一些
    # 墨量：起笔（左上）饱、往右下渐干；再加一点低频的浓淡和收笔处的干
    order = np.clip(((xx - 405) / 1590 * 0.55 + (yy - 487) / 1886 * 0.45), 0, 1)
    load = (1.0 - 0.06 * order) * (0.95 + 0.05 * fbm(H, W, SEED + 80, octaves=3, base=380)) * (1 - 0.30 * dry)
    md = smoothstep(0.15, 0.85, m)
    cov = lerp(d0, d1, hv)
    a_dense = cov * md * lerp(0.90, 1.0, hv) * load
    a_halo = HALO * gaussian(d1, 1.1) * md * (0.45 + 0.55 * hv) * load
    a_soak = WASH * m * load                                    # 很淡的一层，只让字间不发白
    alpha = 1 - (1 - a_dense) * (1 - a_halo) * (1 - a_soak)
    img = img * (1 - alpha[..., None]) + rgbf(INK) * alpha[..., None]

    # ---- 印
    under = np.clip(img / np.maximum(pap, 1e-3), 0, 1) ** 0.3        # 印下面的字透出来一点
    img = img * (1 - sa[..., None]) + (sc * under) * sa[..., None]

    # ---- 光：左上来的掠射光 + 纸面起伏 + 四周微暗
    hgt = fbm(H, W, SEED + 70, octaves=4, base=28) + 0.6 * (fibers(H, W, SEED + 71, n=1800) - 0.5) * 8
    relief = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1) + np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0))
    light = 1.035 - 0.07 * (0.6 * xx / W + 0.4 * yy / H) + 0.03 * np.exp(-((xx / W) ** 2 + (yy / H) ** 2) / 0.12)
    light = light + 0.06 * relief
    img = img * light[..., None]
    img = img * vignette(H, W, strength=0.20, power=2.4, cy=0.48)[..., None]
    save(np.clip(img, 0, 1), out_png, seed=SEED)
    print(f"  共 {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
