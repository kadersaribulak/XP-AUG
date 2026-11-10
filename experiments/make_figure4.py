"""Figure 4: original, preprocessed image, accepted derivative, rejected candidate
and a panel that explains the rejection, for a few source radiographs.

XP-AUG does not store rejected candidates, so the candidates shown here are
generated again with the configuration of an exported run (project_config.json)
and the same quality gate. Seeds are derived deterministically, so the figure
is identical on every run.

Usage (from the XP-AUG_App folder):
    python experiments/make_figure4.py RUN.zip --originals PATH/TO/ORIGINALS --out Figure4.png

RUN.zip       run exported from the Produce page (manifest.csv, project_config.json,
              <class>/<region>/<name>_pre.png)
--originals   folder with the original source images (file names as in manifest.csv,
              e.g. "MPS6_Image 23.png"); searched recursively
--rows        optional, "source|reason" pairs; default is the four rows of the article
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xpaug_core as core  # noqa: E402

DEFAULT_ROWS = [
    "MPS4A_Image 40.png|Content_loss>0.02",
    "MPS6_Image 23.png|near_duplicate_of_source",
    "MPS6_Image 29.png|near_duplicate_of_sibling",
    "MPS4A_Image 17.png|exact_duplicate_of_source",
]
REASON_TEXT = {
    "Content_loss>0.02": "content loss > 0.02",
    "EPI<0.3": "EPI < 0.30",
    "SSIM<0.6": "SSIM < 0.60",
    "AMBE>25.0": "AMBE > 25",
    "near_duplicate_of_source": "near-copy of source",
    "near_duplicate_of_sibling": "near-copy of earlier derivative",
    "exact_duplicate_of_source": "pixel-identical to source",
}
MAX_ATTEMPTS = 500   # candidates tried per source to find the requested rejection
MAX_SIBLINGS = 5     # accepted derivatives kept for the sibling check (= max per source)
DIFF_GAIN = 10       # amplification of difference images


class Run:
    """Read an exported run from a ZIP file or an extracted folder."""

    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.suffix == ".zip" else None
        names = self.zip.namelist() if self.zip else [str(p.relative_to(path)).replace("\\", "/")
                                                      for p in path.rglob("*") if p.is_file()]
        self.names = names
        self.prefix = self._find("manifest.csv").rsplit("manifest.csv", 1)[0]

    def _find(self, suffix: str) -> str:
        hits = [n for n in self.names if n.endswith(suffix)]
        if not hits:
            raise FileNotFoundError(f"{suffix} not found in {self.path}")
        return min(hits, key=len)

    def read(self, name: str) -> bytes:
        full = self.prefix + name
        if self.zip:
            return self.zip.read(full)
        return (self.path / full).read_bytes()

    def manifest(self) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read("manifest.csv")), encoding="utf-8-sig")

    def config(self) -> dict:
        return json.loads(self.read("project_config.json"))

    def image(self, name: str) -> Image.Image:
        return Image.open(io.BytesIO(self.read(name))).convert("L")


def find_original(folder: Path, source: str) -> Path:
    stem = Path(source).stem.lower()
    for p in folder.rglob("*"):
        if p.is_file() and p.stem.lower() == stem:
            return p
    raise FileNotFoundError(f"original image for '{source}' not found in {folder}")


def ops_text(ops: list[str]) -> str:
    if not ops:
        return "no operation triggered"
    out = []
    for o in ops:
        k, v = o.split("=")
        out.append({"gamma": f"γ = {v}", "mult_noise": f"mult. noise σ = {v}",
                    "rotation": f"rotation {v}°"}.get(k, o))
    return ", ".join(out)


def search(pre, mask, aug_cfg, qc, cell, source, base_seed, wanted):
    """Generate candidates until one accepted derivative and the wanted rejection are found."""
    accepted, first_ok = [], None
    for k in range(MAX_ATTEMPTS):
        seed = core.derive_seed(base_seed, cell, Path(source).stem, "figure", k)
        res = core.augment(pre, aug_cfg, seed, mask)
        m = core.augmented_metrics(pre, res, mask)
        thumbs = [core._thumb(a.image) for a in accepted]
        sib = core.min_mad(res.image, thumbs) if thumbs else None
        ok, reason = core.quality_gate(m, qc, sib)
        if ok:
            first_ok = first_ok or (res, m)
            if len(accepted) < MAX_SIBLINGS:
                accepted.append(res)
        elif reason == wanted and first_ok is not None:
            nearest = None
            if wanted == "near_duplicate_of_sibling":
                t = core._thumb(res.image)
                nearest = min(accepted, key=lambda a: np.abs(t - core._thumb(a.image)).mean())
            return first_ok, (res, m, sib, nearest)
    raise RuntimeError(f"no '{wanted}' rejection found for {source} in {MAX_ATTEMPTS} attempts")


def amplified(a, b) -> Image.Image:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) * DIFF_GAIN
    return Image.fromarray(np.clip(d, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--originals", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("Figure4.png"))
    ap.add_argument("--rows", nargs="+", default=DEFAULT_ROWS)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    run = Run(args.run)
    man, cfg = run.manifest(), run.config()
    qc, base_seed = cfg["quality_gate"], int(cfg.get("seed", 42))
    originals = man[man.type == "original"].set_index("source")

    fig, axes = plt.subplots(len(args.rows), 5, figsize=(15, 3.55 * len(args.rows)), squeeze=False)
    for i, spec in enumerate(args.rows):
        source, wanted = spec.split("|")
        rec = originals.loc[source]
        label, region = rec["label"], rec["region"]
        cell = f"{label}|{region}"
        pre = run.image(rec["output"])
        mask = core.foreground_mask(pre)
        aug_cfg = core.region_cfg(cfg["augmentation"], cfg.get("region_overrides", {}),
                                  region, cfg.get("region_aware", True))
        (ra, ma), (rr, mr, sib, nearest) = search(pre, mask, aug_cfg, qc, cell, source, base_seed, wanted)
        orig = Image.open(find_original(args.originals, source)).convert("L")

        if wanted == "Content_loss>0.02":
            value = f"content loss {mr['Content_loss']:.3f}"
            diag, diag_title = pre, "Source with output frame (red);\nanatomy outside it is lost"
        elif wanted == "near_duplicate_of_sibling":
            value = f"MAD to earlier derivative {sib:.2f}"
            diag = amplified(core._thumb(rr.image), core._thumb(nearest.image)).resize(pre.size, Image.NEAREST)
            diag_title = f"|rejected − earlier derivative| × {DIFF_GAIN}\n(128 × 128 thumbnails, as in the gate)"
        else:
            value = f"source MAD {mr['Source_MAD']:.2f}"
            diag, diag_title = amplified(rr.image, pre), f"|rejected − source| × {DIFF_GAIN}"

        panels = [
            (orig, f"Original ({orig.size[0]} × {orig.size[1]} px)\n{label.replace('_', ' ')}, {region.lower()}"),
            (pre, f"Preprocessed ({pre.size[0]} × {pre.size[1]} px)"),
            (ra.image, f"Accepted derivative\n{ops_text(ra.ops)}\nsource MAD {ma['Source_MAD']:.1f}"),
            (rr.image, f"Rejected: {REASON_TEXT.get(wanted, wanted)}\n{ops_text(rr.ops)}\n{value}"),
            (diag, diag_title),
        ]
        for j, (im, title) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_axis_off()
            ax.set_title(title, fontsize=9, loc="left")

        if wanted == "Content_loss>0.02":
            # Output frame mapped back onto the source: rotate the frame by the
            # opposite angle around the image centre.
            angle = [float(o.split("=")[1]) for o in rr.ops if o.startswith("rotation=")]
            if angle:
                w, h = pre.size
                th = -np.deg2rad(angle[0])
                c = np.array([(w - 1) / 2, (h - 1) / 2])
                corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [0, 0]], float) - c
                rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                q = corners @ rot.T + c
                axes[i, 4].plot(q[:, 0], q[:, 1], color="red", lw=1.5)
                axes[i, 4].set_xlim(0, w)
                axes[i, 4].set_ylim(h, 0)

        axes[i, 0].text(-0.08, 1.0, chr(ord("a") + i) + ")", transform=axes[i, 0].transAxes,
                        fontsize=14, fontweight="bold", va="top", ha="right")
        print(f"{chr(ord('a') + i)}) {source}: accepted [{ops_text(ra.ops)}], "
              f"rejected {wanted} [{ops_text(rr.ops)}], {value}")

    plt.tight_layout(h_pad=2.5)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
