"""
XP-AUG core — pure image-processing, quality-control and planning functions.

This module has no Streamlit dependency so that every function can be
unit-tested and reused from scripts.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

try:
    import pywt
    WAVELET_AVAILABLE = True
except Exception:  # pragma: no cover
    pywt = None
    WAVELET_AVAILABLE = False

__version__ = "4.0.0"

# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------
DEFAULT_PRE = {
    "resize": 768, "resize_mode": "Stretch", "padding_value": 0,
    "normalization": "Percentile", "minmax_low": 0, "minmax_high": 255,
    "p_low": 2.0, "p_high": 98.0,
    "clahe": True, "clahe_clip": 2.0, "clahe_grid": 8,
    "hist_eq": False,
    "bilateral": False, "bilateral_d": 5, "bilateral_sigma_color": 50, "bilateral_sigma_space": 50,
    "median": False, "median_ksize": 3,
    "wavelet": False, "wavelet_family": "db1", "wavelet_level": 2, "wavelet_threshold": "soft",
    "homomorphic": False, "homo_gamma_l": 0.5, "homo_gamma_h": 1.8, "homo_cutoff": 30.0,
    # Morphological contrast enhancement: I + TopHat(I) - BlackHat(I)
    "tophat": False, "tophat_kernel": 11,
    "blackhat": False, "blackhat_kernel": 11,
    "unsharp": False, "unsharp_radius": 1.2, "unsharp_percent": 110,
    "gamma_pre": False, "gamma_pre_value": 1.0,
}

DEFAULT_AUG = {
    "gamma": True, "gamma_min": 0.85, "gamma_max": 1.15, "gamma_p": 0.65,
    "brightness": False, "brightness_limit": 0.04, "brightness_p": 0.25,
    "contrast": False, "contrast_limit": 0.10, "contrast_p": 0.25,
    "motion_blur": False, "motion_blur_kernel": 3, "motion_blur_p": 0.20,
    "mult_noise": True, "mult_sigma": 0.05, "mult_p": 0.50,
    "gaussian": False, "gauss_std": 2.0, "gauss_p": 0.30,
    "poisson": False, "poisson_strength": 0.20, "poisson_p": 0.30,
    "rotation": True, "rotation_deg": 8.0, "rotation_p": 0.50,
    "shift": False, "shift_pct": 1.5, "shift_p": 0.30,
    "scale": False, "scale_pct": 2.0, "scale_p": 0.20,
}

# Parameters that may be overridden per region (geometric tolerance differs by anatomy)
REGION_OVERRIDE_KEYS = ["rotation_deg", "shift_pct", "scale_pct"]

DEFAULT_QC = {
    "enabled": True,
    "min_ssim": 0.60,            # structural fidelity to the preprocessed source
    "min_epi": 0.30,             # edge preservation
    "max_ambe": 25.0,            # mean brightness shift (grey levels)
    "reject_exact_duplicates": True,
    "min_mad_to_siblings": 1.0,  # mean absolute grey-level difference to earlier outputs of the same source
    "max_content_loss": 0.02,    # share of foreground pushed out of frame by geometry
    "max_attempts": 5,
}

REGION_KEYWORDS = {
    "spine": ["spine", "vertebra", "lumbar", "thoracic", "cervical"],
    "pelvis_hip": ["pelvis", "hip"],
    "arm": ["arm", "humerus", "elbow", "forearm"],
    "chest": ["chest", "thorax", "lung", "rib"],
    "leg": ["leg", "knee", "tibia", "fibula"],
    "hand": ["hand", "wrist", "finger"],
    "foot": ["foot", "ankle", "toe"],
}

METRIC_COLUMNS = [
    "SSIM", "PSNR_dB", "AMBE", "LCR", "EPI",
    "Entropy_ref", "Entropy_aug", "EME_ref", "EME_aug",
    "dHash_distance", "Exact_duplicate",
    "Content_loss", "Fill_fraction", "Sibling_MAD",
]

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("L")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def derive_seed(*parts) -> int:
    """Deterministic 32-bit seed derived from arbitrary identifiers.

    Every generated image receives its own seed built from the global seed,
    the balancing cell, the source file and the attempt index, so outputs are
    reproducible but not correlated across regions or sources.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def safe_name(text) -> str:
    """Make a label or region usable as a folder name (e.g. 'Pelvis/Hip' -> 'Pelvis_Hip')."""
    t = re.sub(r'[\\/:*?"<>|]+', "_", str(text)).strip().strip(".")
    return t or "unnamed"


def output_path(rec: dict, suffix: str) -> str:
    stem = rec["name"].rsplit(".", 1)[0]
    return f"{safe_name(rec['label'] or 'unlabeled')}/{safe_name(rec['region'])}/{stem}_{suffix}.png"


def pixel_hash(data: bytes) -> str:
    """Hash of decoded grayscale pixel content (detects identical images saved as different files)."""
    im = pil_from_bytes(data)
    return hashlib.sha256(np.asarray(im).tobytes() + str(im.size).encode()).hexdigest()


def guess_region(name: str) -> str:
    n = name.lower()
    for region, keys in REGION_KEYWORDS.items():
        if any(k in n for k in keys):
            return region
    return "unassigned"


# ---------------------------------------------------------------------------
# Quality indicators
# ---------------------------------------------------------------------------
def dhash(img: Image.Image, size: int = 8) -> int:
    a = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    n = 0
    for b in bits:
        n = (n << 1) | int(b)
    return n


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def entropy(img) -> float:
    a = np.asarray(img, dtype=np.uint8)
    h = np.bincount(a.ravel(), minlength=256).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def ambe(ref, aug) -> float:
    """Absolute mean brightness error (Chen & Ramli, 2003)."""
    return float(abs(np.asarray(ref, np.float64).mean() - np.asarray(aug, np.float64).mean()))


def _block_iter(a: np.ndarray, blocks: int):
    h, w = a.shape
    bh, bw = max(1, h // blocks), max(1, w // blocks)
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            yield a[y:y + bh, x:x + bw]


def local_contrast(img, blocks: int = 8) -> float:
    a = np.asarray(img, dtype=np.float64)
    vals = [(b.max() - b.min()) / (b.max() + b.min() + 1e-6) for b in _block_iter(a, blocks)]
    return float(np.mean(vals)) if vals else 0.0


def lcr(ref, aug, blocks: int = 8) -> float:
    """Local contrast ratio: mean block-wise Michelson contrast of output / source."""
    rc = local_contrast(ref, blocks)
    return float("nan") if rc < 1e-12 else float(local_contrast(aug, blocks) / rc)


def epi(ref, aug) -> float:
    """Edge preservation index (Sattar et al., 1997): correlation of the
    mean-removed Laplacian (high-pass) responses of source and output."""
    r = cv2.Laplacian(np.asarray(ref, np.float64), cv2.CV_64F).ravel()
    a = cv2.Laplacian(np.asarray(aug, np.float64), cv2.CV_64F).ravel()
    if r.std() < 1e-8 or a.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(r, a)[0, 1])


def psnr(ref, aug) -> float:
    a = np.asarray(ref, np.float64)
    b = np.asarray(aug, np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


def ssim(ref, aug, data_range: float = 255.0) -> float:
    """Mean SSIM with an 11x11 Gaussian window (sigma = 1.5), as in
    Wang et al. (2004). Implemented with OpenCV to avoid extra dependencies."""
    x = np.asarray(ref, np.float64)
    y = np.asarray(aug, np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sxx + syy + c2)
    smap = num / den
    pad = 5  # discard borders affected by the window
    if smap.shape[0] > 2 * pad and smap.shape[1] > 2 * pad:
        smap = smap[pad:-pad, pad:-pad]
    return float(smap.mean())


def eme(img, blocks: int = 8) -> float:
    """Measure of enhancement, EME (Agaian et al., 2001)."""
    a = np.asarray(img, dtype=np.float64)
    vals = [20.0 * np.log10((b.max() + 1.0) / (b.min() + 1.0)) for b in _block_iter(a, blocks)]
    return float(np.mean(vals)) if vals else 0.0


FIDELITY_COLUMNS = ["SSIM", "PSNR_dB", "AMBE", "LCR", "EPI"]


def geometric_metrics(mask: Image.Image | None, mask_warped: Image.Image | None, coverage: Image.Image) -> dict:
    """Content loss: share of source foreground pixels pushed outside the frame.
    Fill fraction: share of output pixels that are padding introduced by geometry."""
    fill = 1.0 - float((np.asarray(coverage) > 0).mean())
    if mask is None or mask_warped is None:
        return {"Content_loss": float("nan"), "Fill_fraction": fill}
    src = float((np.asarray(mask) > 0).sum())
    kept = float((np.asarray(mask_warped) > 0).sum())
    loss = 0.0 if src == 0 else max(0.0, 1.0 - kept / src)
    return {"Content_loss": loss, "Fill_fraction": fill}


def metrics(ref, aug, dup_ref=None) -> dict:
    """Quality indicators of `aug` relative to `ref`.

    For augmented images, `ref` should be the source warped by the same
    geometric transform (see `augment`), so that fidelity indicators measure
    photometric degradation rather than spatial displacement. Redundancy
    indicators (dHash distance, exact duplicate) are always computed against
    `dup_ref` (the unwarped source); it defaults to `ref`.
    """
    ref = ref.convert("L")
    aug = aug.convert("L")
    if ref.size != aug.size:
        ref = ref.resize(aug.size, Image.Resampling.LANCZOS)
    dup = ref if dup_ref is None else dup_ref.convert("L")
    if dup.size != aug.size:
        dup = dup.resize(aug.size, Image.Resampling.LANCZOS)
    return {
        "SSIM": ssim(ref, aug),
        "PSNR_dB": psnr(ref, aug),
        "AMBE": ambe(ref, aug),
        "LCR": lcr(ref, aug),
        "EPI": epi(ref, aug),
        "Entropy_ref": entropy(ref),
        "Entropy_aug": entropy(aug),
        "EME_ref": eme(ref),
        "EME_aug": eme(aug),
        "dHash_distance": hamming(dhash(dup), dhash(aug)),
        "Exact_duplicate": bool(np.array_equal(np.asarray(dup), np.asarray(aug))),
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _odd(k: int, minimum: int = 3) -> int:
    k = max(minimum, int(k))
    return k if k % 2 else k + 1


def resize_image(img: Image.Image, size: int, mode: str = "Stretch", padding_value: int = 0) -> Image.Image:
    size = int(size)
    if mode == "Stretch":
        return img.resize((size, size), Image.Resampling.LANCZOS)
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), int(padding_value))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def wavelet_denoise_uint8(arr, family="db1", level=2, mode="soft"):
    """Wavelet shrinkage with the universal threshold (Donoho & Johnstone, 1994)."""
    if not WAVELET_AVAILABLE:
        return arr
    x = arr.astype(np.float64)
    coeffs = pywt.wavedec2(x, wavelet=family, level=int(level))
    cD = coeffs[-1][2]
    sigma = float(np.median(np.abs(cD)) / 0.6745) if cD.size else 0.0
    thr = sigma * np.sqrt(2 * np.log(max(arr.size, 2)))
    new = [coeffs[0]] + [tuple(pywt.threshold(c, thr, mode=mode) for c in d) for d in coeffs[1:]]
    rec = pywt.waverec2(new, wavelet=family)[: arr.shape[0], : arr.shape[1]]
    return np.uint8(np.clip(rec, 0, 255))


def homomorphic_filter_uint8(arr, gamma_l=0.5, gamma_h=1.8, cutoff=30.0):
    x = np.log(arr.astype(np.float64) + 1.0)
    rows, cols = arr.shape
    y, xg = np.ogrid[:rows, :cols]
    d2 = (y - rows / 2.0) ** 2 + (xg - cols / 2.0) ** 2
    H = (gamma_h - gamma_l) * (1 - np.exp(-d2 / (2 * cutoff * cutoff))) + gamma_l
    F = np.fft.fftshift(np.fft.fft2(x))
    out = np.exp(np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))) - 1.0
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(np.clip(out, 0, 255))


def morphological_enhance(arr, use_tophat=True, tophat_k=11, use_blackhat=True, blackhat_k=11):
    """Morphological contrast enhancement: I + WTH(I) - BTH(I).

    Both transforms are computed from the same input image; the white
    top-hat adds small bright structures and the black-hat subtracts small
    dark structures, so the anatomy of the original image is preserved.
    """
    base = arr.astype(np.int32)
    out = base.copy()
    if use_tophat:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(tophat_k), _odd(tophat_k)))
        out += cv2.morphologyEx(arr, cv2.MORPH_TOPHAT, k).astype(np.int32)
    if use_blackhat:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(blackhat_k), _odd(blackhat_k)))
        out -= cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, k).astype(np.int32)
    return np.uint8(np.clip(out, 0, 255))


def preprocess(img: Image.Image, c: dict) -> Image.Image:
    """Deterministic preprocessing chain, applied identically to every image.

    Order: resize -> denoise (bilateral, median, wavelet) -> homomorphic ->
    global HE -> morphological enhancement -> CLAHE -> intensity
    normalization -> unsharp masking -> gamma.
    """
    out = img.convert("L")
    if c.get("resize"):
        out = resize_image(out, c["resize"], c.get("resize_mode", "Stretch"), c.get("padding_value", 0))
    a = np.asarray(out, dtype=np.uint8)
    if c.get("bilateral"):
        a = cv2.bilateralFilter(a, int(c["bilateral_d"]), float(c["bilateral_sigma_color"]), float(c["bilateral_sigma_space"]))
    if c.get("median"):
        a = cv2.medianBlur(a, _odd(c["median_ksize"]))
    if c.get("wavelet"):
        a = wavelet_denoise_uint8(a, c.get("wavelet_family", "db1"), c.get("wavelet_level", 2), c.get("wavelet_threshold", "soft"))
    if c.get("homomorphic"):
        a = homomorphic_filter_uint8(a, c.get("homo_gamma_l", 0.5), c.get("homo_gamma_h", 1.8), c.get("homo_cutoff", 30.0))
    if c.get("hist_eq"):
        a = cv2.equalizeHist(a)
    if c.get("tophat") or c.get("blackhat"):
        a = morphological_enhance(a, bool(c.get("tophat")), c.get("tophat_kernel", 11),
                                  bool(c.get("blackhat")), c.get("blackhat_kernel", 11))
    if c.get("clahe"):
        g = int(c["clahe_grid"])
        a = cv2.createCLAHE(float(c["clahe_clip"]), (g, g)).apply(a)
    norm = c.get("normalization", "None")
    if norm == "Min-Max":
        lo, hi = float(a.min()), float(a.max())
        if hi > lo:
            a = ((a - lo) / (hi - lo) * (c["minmax_high"] - c["minmax_low"]) + c["minmax_low"]).clip(0, 255).astype(np.uint8)
    elif norm == "Percentile":
        lo, hi = np.percentile(a, (float(c["p_low"]), float(c["p_high"])))
        if hi > lo:
            a = np.clip((a.astype(np.float64) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    out = Image.fromarray(a)
    if c.get("unsharp"):
        out = out.filter(ImageFilter.UnsharpMask(radius=float(c["unsharp_radius"]), percent=int(c["unsharp_percent"]), threshold=3))
    if c.get("gamma_pre"):
        arr = np.asarray(out, dtype=np.float64) / 255.0
        out = Image.fromarray(np.uint8(np.clip(np.power(arr, float(c["gamma_pre_value"])) * 255, 0, 255)))
    return out


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
PHOTOMETRIC_OPS = ("gamma", "brightness", "contrast", "motion_blur", "mult_noise", "gaussian", "poisson")
GEOMETRIC_OPS = ("rotation", "shift", "scale")


@dataclass
class AugResult:
    image: Image.Image          # augmented output
    ops: list                   # operations actually applied, e.g. "rotation=3.21"
    ref: Image.Image            # source warped by the same geometry (no photometric change)
    mask: Image.Image | None    # foreground mask warped by the same geometry
    coverage: Image.Image       # valid-pixel map (255 inside the original frame)

    @property
    def photometric(self) -> bool:
        return any(o.split("=")[0] in PHOTOMETRIC_OPS for o in self.ops)

    @property
    def geometric(self) -> bool:
        return any(o.split("=")[0] in GEOMETRIC_OPS for o in self.ops)


def foreground_mask(img: Image.Image) -> Image.Image:
    """Binary anatomy mask of a preprocessed radiograph (Otsu threshold)."""
    a = np.asarray(img.convert("L"), dtype=np.uint8)
    _, m = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(m)


def augment(img: Image.Image, c: dict, seed: int, mask: Image.Image | None = None) -> AugResult:
    """Stochastic, probability-controlled augmentation chain.

    Photometric and noise operations are applied first, geometric operations
    last. Every geometric transform is applied identically to three companion
    images: the photometrically untouched source (`ref`), the foreground mask
    and a coverage map. They allow photometric fidelity and geometric content
    loss to be measured separately (see `metrics`).
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    applied: list[str] = []

    def hit(flag, p):
        return c.get(flag) and rng.random() < float(c.get(p, 0))

    if hit("gamma", "gamma_p"):
        g = float(rng.uniform(c["gamma_min"], c["gamma_max"]))
        a = np.asarray(out, np.float64) / 255.0
        out = Image.fromarray(np.uint8(np.clip(np.power(a, g) * 255, 0, 255)))
        applied.append(f"gamma={g:.3f}")
    if hit("brightness", "brightness_p"):
        f = float(rng.uniform(1 - c["brightness_limit"], 1 + c["brightness_limit"]))
        out = ImageEnhance.Brightness(out).enhance(f)
        applied.append(f"brightness={f:.3f}")
    if hit("contrast", "contrast_p"):
        f = float(rng.uniform(1 - c["contrast_limit"], 1 + c["contrast_limit"]))
        out = ImageEnhance.Contrast(out).enhance(f)
        applied.append(f"contrast={f:.3f}")
    if hit("motion_blur", "motion_blur_p"):
        k = _odd(c["motion_blur_kernel"])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        out = Image.fromarray(cv2.filter2D(np.asarray(out, np.uint8), -1, kernel))
        applied.append(f"motion_blur=k{k}")
    if hit("mult_noise", "mult_p"):
        a = np.asarray(out, np.float64)
        out = Image.fromarray(np.uint8(np.clip(a * rng.normal(1, float(c["mult_sigma"]), a.shape), 0, 255)))
        applied.append(f"mult_noise={c['mult_sigma']}")
    if hit("gaussian", "gauss_p"):
        a = np.asarray(out, np.float64)
        out = Image.fromarray(np.uint8(np.clip(a + rng.normal(0, float(c["gauss_std"]), a.shape), 0, 255)))
        applied.append(f"gaussian={c['gauss_std']}")
    if hit("poisson", "poisson_p"):
        a = np.asarray(out, np.float64) / 255.0
        levels = max(10.0, 50.0 / max(float(c["poisson_strength"]), 0.01))
        out = Image.fromarray(np.uint8(np.clip(rng.poisson(np.clip(a, 0, 1) * levels) / levels * 255, 0, 255)))
        applied.append(f"poisson={c['poisson_strength']}")
    ref = img.copy()
    cov = Image.new("L", img.size, 255)
    msk = mask.copy() if mask is not None else None

    def geo(fn_img, fn_mask):
        nonlocal out, ref, cov, msk
        out, ref = fn_img(out), fn_img(ref)
        cov = fn_mask(cov)
        if msk is not None:
            msk = fn_mask(msk)

    if hit("rotation", "rotation_p") and c["rotation_deg"] > 0:
        deg = float(rng.uniform(-c["rotation_deg"], c["rotation_deg"]))
        geo(lambda im: im.rotate(deg, resample=Image.Resampling.BILINEAR, fillcolor=0),
            lambda im: im.rotate(deg, resample=Image.Resampling.NEAREST, fillcolor=0))
        applied.append(f"rotation={deg:.2f}")
    if hit("shift", "shift_p") and c["shift_pct"] > 0:
        W, H = out.size
        dx, dy = int(W * c["shift_pct"] / 100), int(H * c["shift_pct"] / 100)
        tx = int(rng.integers(-dx, dx + 1)) if dx else 0
        ty = int(rng.integers(-dy, dy + 1)) if dy else 0
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        warp = lambda interp: (lambda im: Image.fromarray(cv2.warpAffine(np.asarray(im), M, (W, H), flags=interp, borderValue=0)))
        geo(warp(cv2.INTER_LINEAR), warp(cv2.INTER_NEAREST))
        applied.append(f"shift=({tx},{ty})")
    if hit("scale", "scale_p") and c["scale_pct"] > 0:
        f = float(rng.uniform(1 - c["scale_pct"] / 100, 1 + c["scale_pct"] / 100))
        w, h = out.size
        nw, nh = max(1, int(w * f)), max(1, int(h * f))

        def scaler(resample):
            def _scale(im):
                t = im.resize((nw, nh), resample)
                if nw > w or nh > h:  # zoom in: centre crop
                    l, u = (nw - w) // 2, (nh - h) // 2
                    return t.crop((l, u, l + w, u + h))
                z = Image.new("L", (w, h), 0)
                z.paste(t, ((w - nw) // 2, (h - nh) // 2))
                return z
            return _scale
        geo(scaler(Image.Resampling.BILINEAR), scaler(Image.Resampling.NEAREST))
        applied.append(f"scale={f:.3f}")
    return AugResult(out, applied, ref, msk, cov)


def augmented_metrics(src: Image.Image, res: AugResult, src_mask: Image.Image | None) -> dict:
    """All indicators for one augmented image.

    Fidelity indicators (SSIM, PSNR, AMBE, LCR, EPI) are computed against the
    geometrically aligned source and are reported as NaN when no photometric
    operation was applied: in that case they would be trivially perfect and
    carry no information. Geometric operations are assessed by content loss
    and fill fraction instead. Redundancy is always assessed against the
    unwarped source.
    """
    m = metrics(res.ref, res.image, dup_ref=src)
    if not res.photometric:
        for k in FIDELITY_COLUMNS:
            m[k] = float("nan")
    m.update(geometric_metrics(src_mask, res.mask, res.coverage))
    return m


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _thumb(img: Image.Image, size: int = 128) -> np.ndarray:
    return np.asarray(img.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def min_mad(img: Image.Image, siblings: list) -> float | None:
    """Smallest mean absolute grey-level difference to previously accepted
    outputs of the same source (computed on 128x128 thumbnails)."""
    if not siblings:
        return None
    t = _thumb(img)
    return float(min(np.abs(t - s).mean() for s in siblings))


def quality_gate(m: dict, qc: dict, sibling_min_mad: float | None) -> tuple[bool, str]:
    """Accept or reject a generated image. Returns (accepted, reason)."""
    if not qc.get("enabled", True):
        return True, ""
    if qc.get("reject_exact_duplicates", True) and m["Exact_duplicate"]:
        return False, "exact_duplicate_of_source"
    if not _nan(m["SSIM"]) and m["SSIM"] < qc["min_ssim"]:
        return False, f"SSIM<{qc['min_ssim']}"
    if not _nan(m["EPI"]) and m["EPI"] < qc["min_epi"]:
        return False, f"EPI<{qc['min_epi']}"
    if not _nan(m["AMBE"]) and m["AMBE"] > qc["max_ambe"]:
        return False, f"AMBE>{qc['max_ambe']}"
    if not _nan(m.get("Content_loss")) and m["Content_loss"] > qc.get("max_content_loss", 1.0):
        return False, f"Content_loss>{qc['max_content_loss']}"
    if sibling_min_mad is not None and sibling_min_mad < qc.get("min_mad_to_siblings", 0.0):
        return False, "near_duplicate_of_sibling"
    return True, ""


# ---------------------------------------------------------------------------
# Splitting and balancing
# ---------------------------------------------------------------------------
def cell_key(rec: dict, unit: str) -> str:
    if unit == "Class x Region":
        return f"{rec['label']}|{rec['region']}"
    if unit == "Class":
        return rec["label"]
    return rec["region"]


def balancing_plan(records: list[dict], unit: str, mode: str, fixed_target: int = 0) -> pd.DataFrame:
    """Number of images to generate per balancing cell."""
    if not records:
        return pd.DataFrame(columns=["cell", "count", "target", "to_generate"])
    counts = pd.Series([cell_key(r, unit) for r in records]).value_counts()
    plan = counts.rename_axis("cell").reset_index(name="count").sort_values("cell").reset_index(drop=True)
    if mode == "Match largest cell":
        target = int(plan["count"].max())
        plan["target"] = np.maximum(plan["count"], target)
    elif mode == "Fixed target":
        plan["target"] = np.maximum(plan["count"], int(fixed_target))
    else:
        plan["target"] = plan["count"]
    plan["to_generate"] = plan["target"] - plan["count"]
    return plan


def region_cfg(aug: dict, overrides: dict, region: str, region_aware: bool) -> dict:
    cfg = dict(aug)
    if region_aware:
        cfg.update({k: v for k, v in overrides.get(region, {}).items() if k in REGION_OVERRIDE_KEYS})
    return cfg


@dataclass
class GenerationResult:
    outputs: list = field(default_factory=list)   # (archive_path, png_bytes)
    manifest: list = field(default_factory=list)  # accepted rows
    rejected: list = field(default_factory=list)  # rejected attempts
    shortfall: dict = field(default_factory=dict) # cell -> images that could not be produced


def generate(records, pre_cache, plan, unit, aug, overrides, region_aware, qc, base_seed=42, progress=None,
             max_per_source: int = 0):
    """Generate augmented training images according to the balancing plan.

    records   : list of dicts with name, region, label and optional caption
                (the caption of the source is inherited by every augmented image)
    pre_cache : {name: preprocessed PIL image}
    """
    res = GenerationResult()
    total = int(plan["to_generate"].sum()) if len(plan) else 0
    done = 0
    for _, row in plan.iterrows():
        need = int(row.to_generate)
        if need <= 0:
            continue
        srcs = sorted([r for r in records if cell_key(r, unit) == row.cell], key=lambda r: r["name"])
        siblings: dict[str, list] = {s["name"]: [] for s in srcs}
        masks = {s["name"]: foreground_mask(pre_cache[s["name"]]) for s in srcs}
        per_source_idx: dict[str, int] = {s["name"]: 0 for s in srcs}
        accepted_per_source: dict[str, int] = {s["name"]: 0 for s in srcs}
        produced = 0
        i = 0
        max_total_attempts = need * max(1, int(qc.get("max_attempts", 5))) if qc.get("enabled", True) else need
        attempts = 0
        while produced < need and attempts < max_total_attempts:
            if max_per_source:
                open_srcs = [x for x in srcs if accepted_per_source[x["name"]] < max_per_source]
                if not open_srcs:
                    break
            else:
                open_srcs = srcs
            r = open_srcs[i % len(open_srcs)]  # round-robin over sources that are still open
            i += 1
            attempts += 1
            per_source_idx[r["name"]] += 1
            k = per_source_idx[r["name"]]
            seed = derive_seed(base_seed, row.cell, r["name"], k)
            pre = pre_cache[r["name"]]
            cfg = region_cfg(aug, overrides, r["region"], region_aware)
            ar = augment(pre, cfg, seed, masks[r["name"]])
            out, ops = ar.image, ar.ops
            m = augmented_metrics(pre, ar, masks[r["name"]])
            h = dhash(out)
            sib = min_mad(out, siblings[r["name"]])
            m["Sibling_MAD"] = sib if sib is not None else float("nan")
            ok, reason = quality_gate(m, qc, sib)
            if not ok:
                res.rejected.append({"source": r["name"], "cell": row.cell, "seed": seed, "reason": reason,
                                     "operations": ";".join(ops),
                                     **{kk: m[kk] for kk in ("SSIM", "EPI", "AMBE", "Content_loss", "dHash_distance")}})
                continue
            siblings[r["name"]].append(_thumb(out))
            accepted_per_source[r["name"]] += 1
            data = png_bytes(out)
            path = output_path(r, f"aug_{k:03d}")
            res.outputs.append((path, data))
            res.manifest.append({
                "output": path, "type": "augmented", "source": r["name"],
                "label": r["label"], "region": r["region"], "caption": r.get("caption", ""), "seed": seed,
                "operations": ";".join(ops), "photometric": ar.photometric, "geometric": ar.geometric, **m, "sha256": sha256(data), "dhash": hex(h),
            })
            produced += 1
            done += 1
            if progress:
                progress(min(done / max(total, 1), 1.0))
        if produced < need:
            res.shortfall[row.cell] = need - produced
    return res
