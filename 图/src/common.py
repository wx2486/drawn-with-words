"""「用字画的」这组图的公共工具。

做法：文字一律交给 headless Chrome 排（SVG + textPath，多语种字形和连写都由浏览器负责），
渲染成白字黑底的覆盖率蒙版；光、颜色、纸、辉光、颗粒都在 numpy 里合成。
"""
import os
import subprocess
import tempfile
import time
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CHROME = os.environ.get("CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
ROOT = Path(__file__).resolve().parent.parent          # 表达/图
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)

# 字体栈：中文用思源宋体，西文用 Georgia / Cambria，其余脚本交给系统回退（Nirmala UI 等）
SERIF = "'Noto Serif SC', Georgia, Cambria, 'Times New Roman', 'Nirmala UI', 'Segoe UI', serif"
SANS = "'Noto Sans SC', 'Segoe UI', 'Nirmala UI', sans-serif"

W, H = 2400, 3200      # 成图尺寸（3:4）


def esc(s):
    return escape(s, quote=True)


def chrome_shot(html, w, h, name="shot", timeout=600):
    """把一段 HTML 用 headless Chrome 截成 w×h 的 PNG，返回 uint8 RGB 数组。"""
    WORK.mkdir(exist_ok=True)
    html_path = WORK / f"{name}.html"
    png_path = WORK / f"{name}.png"
    html_path.write_text(html, encoding="utf-8")
    if png_path.exists():
        png_path.unlink()
    with tempfile.TemporaryDirectory(prefix="chrome-ud-") as ud:
        cmd = [
            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--user-data-dir={ud}", "--force-device-scale-factor=1",
            "--default-background-color=000000ff",
            "--virtual-time-budget=20000",
            f"--screenshot={png_path}", f"--window-size={w},{h}",
            html_path.as_uri(),
        ]
        t0 = time.time()
        subprocess.run(cmd, check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not png_path.exists():
        raise RuntimeError(f"Chrome 没有输出 {png_path}")
    img = np.asarray(Image.open(png_path).convert("RGB"))
    if img.shape[0] != h or img.shape[1] != w:
        raise RuntimeError(f"截图尺寸 {img.shape[1]}×{img.shape[0]}，期望 {w}×{h}")
    print(f"  chrome {name}: {w}×{h} {time.time() - t0:.1f}s")
    return img


def svg_mask(body, w=W, h=H, ss=2, name="mask", defs="", css=""):
    """在 w×h 的坐标系里画 SVG（白色 = 有墨），以 ss 倍超采样渲染后按面积缩回，返回 [0,1] float32。"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;padding:0;background:#000;overflow:hidden}}
svg{{display:block}}
text{{font-kerning:normal;text-rendering:geometricPrecision}}
{css}
</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" width="{w * ss}" height="{h * ss}" viewBox="0 0 {w} {h}">
<defs>{defs}</defs>
{body}
</svg></body></html>"""
    img = chrome_shot(html, w * ss, h * ss, name=name)
    m = img.astype(np.float32).mean(axis=2) / 255.0
    if ss != 1:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(m, 0, 1)


# ---------------------------------------------------------------- 纹理与后期

def fbm(h, w, seed=0, octaves=6, base=256.0, persistence=0.5):
    """多倍频值噪声（插值的随机格点），快且够用；返回大约 [-1,1]。"""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    scale = base
    for _ in range(octaves):
        gh, gw = int(h / scale) + 3, int(w / scale) + 3
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (int(gw * scale), int(gh * scale)), interpolation=cv2.INTER_CUBIC)
        oy, ox = rng.integers(0, max(1, int(scale))), rng.integers(0, max(1, int(scale)))
        out += amp * up[oy:oy + h, ox:ox + w]
        total += amp
        amp *= persistence
        scale /= 2.0
        if scale < 1.5:
            break
    return out / total


def fibers(h, w, seed=0, n=2600, length=(20, 90), alpha=0.06):
    """纸纤维：许多细短的弯线，返回 [0,1] 的明暗扰动层（0.5 为中性）。"""
    rng = np.random.default_rng(seed)
    layer = np.full((h, w), 0.5, np.float32)
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        L = rng.uniform(*length)
        a = rng.uniform(0, np.pi)
        bend = rng.normal(0, 0.4)
        pts = []
        for t in np.linspace(0, 1, 8):
            aa = a + bend * (t - 0.5)
            pts.append((x + np.cos(aa) * L * (t - 0.5), y + np.sin(aa) * L * (t - 0.5)))
        pts = np.array(pts, np.float32)
        v = 0.5 + rng.choice([-1, 1]) * alpha * rng.uniform(0.4, 1.0)
        cv2.polylines(layer, [np.round(pts * 4).astype(np.int32)], False, float(v), 1,
                      cv2.LINE_AA, shift=2)
    return cv2.GaussianBlur(layer, (0, 0), 0.6)


def paper(h, w, rgb, seed=0, mottle=0.035, fiber=1.0, grain=0.012):
    """一张纸：底色 + 大尺度斑驳 + 纤维 + 细颗粒。返回 float32 RGB [0,1]。"""
    rgb = np.array(rgb, np.float32) / 255.0 if max(rgb) > 1 else np.array(rgb, np.float32)
    n = fbm(h, w, seed, octaves=7, base=420)
    f = (fibers(h, w, seed + 1) - 0.5) * fiber
    g = np.random.default_rng(seed + 2).standard_normal((h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.7)
    lum = 1.0 + mottle * n + f + grain * g
    return np.clip(rgb[None, None, :] * lum[..., None], 0, 1)


def gaussian(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def bloom(rgb, sigmas=(4, 16, 48), weights=(0.35, 0.25, 0.18), threshold=0.0):
    """把亮部糊开再加回去，模拟镜头和眼睛里的光晕。"""
    src = np.clip(rgb - threshold, 0, None)
    acc = np.zeros_like(rgb)
    for s, wgt in zip(sigmas, weights):
        acc += wgt * gaussian(src, s)
    return rgb + acc


def vignette(h, w, strength=0.25, power=2.2, cx=0.5, cy=0.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx / w - cx) / 0.75
    dy = (yy / h - cy) / 0.75
    r = np.sqrt(dx * dx + dy * dy)
    return (1.0 - strength * np.clip(r, 0, 1.5) ** power).astype(np.float32)


def to_uint8(rgb, seed=0):
    """线性合成结果 → 8 位，加半个色阶的随机抖动避免渐变断层。"""
    rng = np.random.default_rng(seed)
    d = (rng.random(rgb.shape, dtype=np.float32) - 0.5) / 255.0
    return np.clip((rgb + d) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save(rgb, path, seed=0):
    path = Path(path)
    Image.fromarray(to_uint8(rgb, seed)).save(path, optimize=True)
    print(f"saved {path}  {path.stat().st_size / 1e6:.1f} MB")


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def rgbf(hexstr):
    hexstr = hexstr.lstrip("#")
    return np.array([int(hexstr[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


# ---------------------------------------------------------------- 多语种文字

FONT = {
    "zh": "'Noto Serif SC'",
    "ja": "'Noto Serif SC'",
    "lat": "Cambria, Georgia, 'Times New Roman', serif",     # 拉丁、希腊、西里尔
    "he": "'Times New Roman', 'Segoe UI'",
    "fa": "'Times New Roman', 'Segoe UI'",
    "ar": "'Times New Roman', 'Segoe UI'",
    "bn": "'Nirmala UI'",
    "hi": "'Nirmala UI'",
    "ko": "'Malgun Gothic'",
}

# 估算排版宽度用的平均字宽（单位：字号）
_ADV = {"zh": 1.0, "ja": 1.0, "ko": 0.95, "lat": 0.47, "he": 0.5, "fa": 0.42, "ar": 0.42, "bn": 0.6, "hi": 0.6}


def est_width(lang, text, size):
    return len(text) * _ADV.get(lang, 0.55) * size


def tspans(items, sep="  ·  ", sep_lang="lat", extra=""):
    """[(lang, text), ...] → 一串 <tspan>，每段用各自脚本的字体。"""
    out = []
    for k, (lang, t) in enumerate(items):
        if k:
            out.append(f'<tspan font-family="{FONT[sep_lang]}">{esc(sep)}</tspan>')
        out.append(f'<tspan font-family="{FONT[lang]}" {extra}>{esc(t)}</tspan>')
    return "".join(out)


def fill_line(pool, start, length, size, sep="  ·  "):
    """从 pool 的第 start 条起依次取句子，直到估算宽度超过 length。返回 (items, 下一条的序号)。"""
    items, w, k = [], 0.0, start
    while w < length:
        lang, t = pool[k % len(pool)]
        items.append((lang, t))
        w += est_width(lang, t, size) + est_width("lat", sep, size)
        k += 1
    return items, k
