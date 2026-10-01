"""
XP-AUG — Medical X-ray Preparation & Augmentation Studio (Streamlit UI).

Run:  streamlit run app.py
"""
from __future__ import annotations

import copy
import os
import platform
import time
import io
import json
import zipfile
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import xpaug_core as core
from PIL import Image

BASE = Path(__file__).resolve().parent
LOGO = BASE / "assets" / "idea_logo.png"

st.set_page_config(page_title="XP-AUG", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{--navy:#073B8E;--blue:#0B63F6;--ink:#10234A;--muted:#7181A0;--line:#E5ECF6}
.stApp{background:linear-gradient(180deg,#FCFDFF,#F6F9FE);color:var(--ink)}
.block-container{max-width:1450px;padding-top:1rem;padding-bottom:2.5rem}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.hero{padding:.2rem 0 .7rem}
.med-title{font-size:3.45rem;font-weight:850;letter-spacing:-.055em;line-height:1;color:#071D55}
.med-title .aug{color:#1677E8}
.med-sub{font-size:1.04rem;color:#63769A;margin-top:.45rem}
.top-card,.card{background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:0 8px 24px rgba(20,54,110,.05)}
.top-card{padding:17px 20px}.card{padding:20px 22px;margin-bottom:1rem}
.card-title{font-size:.88rem;font-weight:800;color:var(--blue);text-transform:uppercase;letter-spacing:.02em;margin-bottom:12px}
.metric-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:17px 18px;min-height:112px;box-shadow:0 8px 22px rgba(20,54,110,.045)}
.metric-num{font-size:1.7rem;font-weight:800;color:var(--blue)}
.metric-label{font-weight:700;margin-top:2px}.metric-note{font-size:.8rem;color:#7B8CAA;margin-top:3px}
div[data-testid="stButton"] button{border-radius:10px;font-weight:750;min-height:42px}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "page": "Overview", "project_name": "XP-AUG_Project", "records": [], "metadata": None,
    "cols": {"file": None, "region": None, "label": None, "caption": None},
    "pre": copy.deepcopy(core.DEFAULT_PRE), "aug": copy.deepcopy(core.DEFAULT_AUG),
    "qc": copy.deepcopy(core.DEFAULT_QC),
    "region_aware": False, "region_overrides": {},
    "balance_unit": "Class x Region", "target_mode": "Match largest cell", "fixed_target": 30,
    "seed": 42,
    "pre_cache": {}, "pre_cache_key": None,
    "last_manifest": None, "last_rejected": None, "last_summary": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

S = st.session_state
NONE = "(none)"


def records_df() -> pd.DataFrame:
    if not S.records:
        return pd.DataFrame(columns=["filename", "label", "region", "caption"])
    return pd.DataFrame([{"filename": r["name"], "label": r["label"], "region": r["region"],
                          "caption": r.get("caption", "")} for r in S.records])


def get_pre_cache() -> dict:
    """Preprocess each source image once per preprocessing configuration."""
    key = core.sha256(json.dumps(S.pre, sort_keys=True).encode()) + str(len(S.records))
    if S.pre_cache_key != key:
        t = time.perf_counter()
        S.pre_cache = {r["name"]: core.preprocess(core.pil_from_bytes(r["data"]), S.pre) for r in S.records}
        S.pre_time = time.perf_counter() - t  # time to preprocess the whole dataset once
        S.pre_cache_key = key
    return S.pre_cache


def apply_metadata():
    md, c = S.metadata, S.cols
    if md is None or not c["file"]:
        return
    idx = {str(row[c["file"]]).strip(): row for _, row in md.iterrows()}
    for r in S.records:
        row = idx.get(r["name"])
        if row is None:
            continue
        r["matched"] = True
        for fld in ("region", "label", "caption"):
            col = c[fld]
            if col and col != NONE and pd.notna(row[col]):
                val = str(row[col]).strip()
                r[fld] = val


def nav(label, target):
    if st.button(label, width="stretch", type="primary" if S.page == target else "secondary"):
        S.page = target
        st.rerun()


def system_info() -> dict:
    """Basic hardware/software description for reporting runtimes."""
    cpu = platform.processor() or platform.machine()
    ram_gb = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            ram_gb = round(int(f.readline().split()[1]) / 1024 ** 2, 1)
    except OSError:
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
        except Exception:
            pass
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "ram_gb": ram_gb,
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version()}


def metric_card(num, label, note):
    st.markdown(f'<div class="metric-card"><div class="metric-num">{num}</div>'
                f'<div class="metric-label">{label}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True)


INK = "#10234A"
PALETTE = ["#0B63F6", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16", "#64748B"]


def card_title(text):
    st.markdown(f'<div class="card-title" style="margin-top:.6rem">{text}</div>', unsafe_allow_html=True)


def chips(items, color):
    html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin:4px 6px 4px 0;padding:7px 12px;'
        f'border-radius:999px;background:{color}14;border:1px solid {color}44;color:{INK};font-size:.86rem">'
        f'<b style="color:{color}">{i + 1}</b> {name}'
        f'{f"<span style=\'color:#7181A0\'>· {detail}</span>" if detail else ""}</span>'
        f'{"<span style=\'color:#A0AEC0;margin-right:6px\'>→</span>" if i < len(items) - 1 else ""}'
        for i, (name, detail) in enumerate(items))
    st.markdown(f"<div>{html}</div>", unsafe_allow_html=True)


def thumb(rec, size=220):
    im = core.pil_from_bytes(rec["data"])
    im.thumbnail((size, size))
    return im


def image_stats() -> pd.DataFrame:
    """Per-image statistics of the original uploads (cached by content hash)."""
    cache = S.setdefault("img_stats", {})
    rows = []
    for r in S.records:
        if r["sha256"] not in cache:
            raw = Image.open(io.BytesIO(r["data"]))
            w, h, mode = raw.size[0], raw.size[1], raw.mode
            g = core.pil_from_bytes(r["data"])
            g.thumbnail((512, 512))
            a = np.asarray(g, dtype=np.float64)
            p1, p99 = np.percentile(a, (1, 99))
            hist = np.bincount(a.astype(np.uint8).ravel(), minlength=256) / a.size
            cache[r["sha256"]] = {"width": w, "height": h, "mode": mode, "size_kb": len(r["data"]) / 1024,
                                  "mean": a.mean(), "std": a.std(), "entropy": core.entropy(a.astype(np.uint8)),
                                  "dyn_range": p99 - p1, "hist": hist}
        d = {k: v for k, v in cache[r["sha256"]].items() if k != "hist"}
        rows.append({"name": r["name"], "region": r["region"], "label": r["label"], **d})
    return pd.DataFrame(rows)


def intensity_histograms(bins=64) -> pd.DataFrame:
    image_stats()
    out = []
    for reg in sorted({r["region"] for r in S.records}):
        hs = [S.img_stats[r["sha256"]]["hist"] for r in S.records if r["region"] == reg]
        mh = np.mean(hs, axis=0).reshape(bins, -1).sum(axis=1)
        for i, v in enumerate(mh):
            out.append({"region": reg, "grey": int(i * 256 / bins), "share": float(v)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
PAGES = ["Overview", "Dataset", "Preprocess", "Augmentation", "Produce", "Results"]
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width="stretch")
    st.markdown(f"### 🩻 XP-AUG `v{core.__version__}`")
    st.divider()
    for p in PAGES:
        nav(p, p)
    st.divider()
    S.project_name = st.text_input("Project", S.project_name)
    S.seed = int(st.number_input("Global seed", 0, 2**31 - 1, int(S.seed)))

hl, hr = st.columns([2.4, 1])
with hl:
    st.markdown('<div class="hero"><div class="med-title">🩻 XP-<span class="aug">AUG</span></div>'
                '<div class="med-sub">Medical X-ray Preparation & Augmentation Studio</div></div>',
                unsafe_allow_html=True)
with hr:
    df0 = records_df()
    st.markdown(f'<div class="top-card"><b>{S.project_name}</b><br>'
                f'<span>{len(df0)} images · {df0["label"].nunique() if len(df0) else 0} classes · '
                f'{df0["region"].nunique() if len(df0) else 0} regions</span></div>', unsafe_allow_html=True)

page = S.page

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    df = records_df()
    n = len(df)
    if not n:
        st.markdown('<div class="card" style="text-align:center;padding:48px">'
                    '<div style="font-size:3rem">🩻</div>'
                    '<div style="font-size:1.3rem;font-weight:800;margin-top:6px">No dataset loaded yet</div>'
                    '<div style="color:#7181A0;margin-top:6px">Upload radiographs and an optional metadata table on the '
                    '<b>Dataset</b> page to see the dataset profile, image statistics and pipeline overview.</div></div>',
                    unsafe_allow_html=True)
        st.stop()

    stats = image_stats()
    labelled = df["label"].nunique() > 1 or df["label"].iloc[0] != "unlabeled"
    cells = df.groupby(["label", "region"]).size() if labelled else df.groupby("region").size()
    imb = cells.max() / max(cells.min(), 1)
    med_w, med_h = int(stats["width"].median()), int(stats["height"].median())
    matched = sum(r.get("matched", False) for r in S.records)

    m = st.columns(6)
    with m[0]: metric_card(n, "Images", f"{matched} matched to metadata")
    with m[1]: metric_card(df["label"].nunique() if labelled else "–", "Classes", "Diagnostic groups" if labelled else "No class column")
    with m[2]: metric_card(df["region"].nunique(), "Regions", f"{int((df['region'] == 'unassigned').sum())} unassigned")
    with m[3]: metric_card(f"{imb:.1f}×", "Imbalance", "Largest / smallest " + ("cell" if labelled else "region"))
    with m[4]: metric_card(f"{med_w}×{med_h}", "Median size", f"{stats['width'].min()}–{stats['width'].max()} px wide")
    with m[5]: metric_card(stats["mode"].nunique(), "Formats", ", ".join(sorted(stats["mode"].unique()))[:28])

    if (df["region"] == "unassigned").any():
        st.warning(f"{int((df['region'] == 'unassigned').sum())} image(s) have no region. Assign them on the Dataset page.")

    tab_ds, tab_img, tab_pipe, tab_run = st.tabs(["📊 Dataset profile", "🔬 Image statistics", "⚙️ Pipeline", "🚀 Last production run"])

    # ---------------- Dataset profile ----------------
    with tab_ds:
        a, b = st.columns([1, 1.15])
        with a:
            card_title("Class × region")
            ct = df.groupby(["label", "region"]).size().reset_index(name="images")
            base = alt.Chart(ct).encode(
                x=alt.X("region:N", title=None, sort="-y", axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("label:N", title=None))
            heat = base.mark_rect(cornerRadius=4).encode(
                color=alt.Color("images:Q", scale=alt.Scale(scheme="blues"), legend=None),
                tooltip=["label", "region", "images"])
            txt = base.mark_text(fontWeight="bold", fontSize=13).encode(
                text="images:Q",
                color=alt.condition(alt.datum.images > ct["images"].max() * 0.55, alt.value("white"), alt.value(INK)))
            st.altair_chart((heat + txt).properties(height=max(140, 60 * ct["label"].nunique())), width="stretch")
        with b:
            card_title("Images per region")
            st.altair_chart(alt.Chart(ct).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("region:N", title=None, sort="-x"),
                x=alt.X("sum(images):Q", title="Images"),
                color=alt.Color("label:N", title="Class", scale=alt.Scale(range=PALETTE)),
                tooltip=["region", "label", "images"]).properties(height=260), width="stretch")

        card_title("One sample per region")
        regions = sorted(df["region"].unique())
        cols = st.columns(min(len(regions), 6))
        for i, reg in enumerate(regions):
            rec = next(r for r in S.records if r["region"] == reg)
            with cols[i % len(cols)]:
                st.image(thumb(rec), caption=f"{reg} · {int((df['region'] == reg).sum())} img", width="stretch")

    # ---------------- Image statistics ----------------
    with tab_img:
        st.caption("Computed on the original (unprocessed) images. Heterogeneity here is what preprocessing is meant to harmonize.")
        a, b = st.columns(2)
        with a:
            card_title("Acquisition geometry")
            st.altair_chart(alt.Chart(stats).mark_circle(size=90, opacity=0.75).encode(
                x=alt.X("width:Q", title="Width (px)", scale=alt.Scale(zero=False)),
                y=alt.Y("height:Q", title="Height (px)", scale=alt.Scale(zero=False)),
                color=alt.Color("region:N", scale=alt.Scale(range=PALETTE)),
                tooltip=["name", "region", "width", "height", "mode", alt.Tooltip("size_kb:Q", format=".0f")]
            ).properties(height=300).interactive(), width="stretch")
        with b:
            card_title("Brightness vs. contrast")
            st.altair_chart(alt.Chart(stats).mark_circle(size=90, opacity=0.75).encode(
                x=alt.X("mean:Q", title="Mean intensity", scale=alt.Scale(zero=False)),
                y=alt.Y("std:Q", title="Global contrast (SD)", scale=alt.Scale(zero=False)),
                color=alt.Color("region:N", scale=alt.Scale(range=PALETTE)),
                shape=alt.Shape("label:N", title="Class"),
                tooltip=["name", "region", "label", alt.Tooltip("mean:Q", format=".1f"), alt.Tooltip("std:Q", format=".1f")]
            ).properties(height=300).interactive(), width="stretch")

        metric_choice = st.segmented_control("Distribution of", ["Entropy", "Dynamic range", "Mean intensity", "Contrast (SD)"],
                                             default="Entropy")
        col = {"Entropy": "entropy", "Dynamic range": "dyn_range", "Mean intensity": "mean", "Contrast (SD)": "std"}[metric_choice or "Entropy"]
        a, b = st.columns([1.3, 1])
        with a:
            card_title(f"{metric_choice or 'Entropy'} by region")
            st.altair_chart(alt.Chart(stats).mark_boxplot(size=26, median={"color": "white"}).encode(
                x=alt.X("region:N", title=None, axis=alt.Axis(labelAngle=-30)),
                y=alt.Y(f"{col}:Q", title=metric_choice, scale=alt.Scale(zero=False)),
                color=alt.Color("region:N", scale=alt.Scale(range=PALETTE), legend=None)
            ).properties(height=280), width="stretch")
        with b:
            card_title("Overall distribution")
            st.altair_chart(alt.Chart(stats).transform_density(col, as_=[col, "density"], groupby=["label"]).mark_area(
                opacity=0.45, interpolate="monotone").encode(
                x=alt.X(f"{col}:Q", title=metric_choice),
                y=alt.Y("density:Q", title=None, axis=None, stack=None),
                color=alt.Color("label:N", title="Class", scale=alt.Scale(range=PALETTE))
            ).properties(height=280), width="stretch")

        card_title("Mean intensity histogram per region")
        hist = intensity_histograms()
        st.altair_chart(alt.Chart(hist).mark_line(interpolate="monotone", strokeWidth=2).encode(
            x=alt.X("grey:Q", title="Grey level", scale=alt.Scale(domain=[0, 255])),
            y=alt.Y("share:Q", title="Share of pixels", axis=alt.Axis(format="%")),
            color=alt.Color("region:N", scale=alt.Scale(range=PALETTE)),
            tooltip=["region", "grey", alt.Tooltip("share:Q", format=".2%")]
        ).properties(height=240).interactive(bind_y=False), width="stretch")

    # ---------------- Pipeline ----------------
    with tab_pipe:
        pre_steps = [
            ("Resize", f"{S.pre['resize']} px · {S.pre['resize_mode'].split()[0].lower()}", True),
            ("Bilateral", f"d={S.pre['bilateral_d']}", S.pre["bilateral"]),
            ("Median", f"k={S.pre['median_ksize']}", S.pre["median"]),
            ("Wavelet", f"{S.pre['wavelet_family']} · L{S.pre['wavelet_level']}", S.pre["wavelet"]),
            ("Homomorphic", f"γ {S.pre['homo_gamma_l']}–{S.pre['homo_gamma_h']}", S.pre["homomorphic"]),
            ("Global HE", "", S.pre["hist_eq"]),
            ("Morphology", "+TH" * S.pre["tophat"] + " −BH" * S.pre["blackhat"], S.pre["tophat"] or S.pre["blackhat"]),
            ("CLAHE", f"clip {S.pre['clahe_clip']} · {S.pre['clahe_grid']}×{S.pre['clahe_grid']}", S.pre["clahe"]),
            ("Normalize", S.pre["normalization"], S.pre["normalization"] != "None"),
            ("Unsharp", f"{S.pre['unsharp_percent']}%", S.pre["unsharp"]),
            ("Gamma", f"γ={S.pre['gamma_pre_value']}", S.pre["gamma_pre"]),
        ]
        card_title("Preprocessing chain (applied in this order)")
        chips([(n_, d_) for n_, d_, on in pre_steps if on], "#0B63F6")
        a_ = S.aug
        aug_ops = [
            ("Gamma", "gamma", f"{a_['gamma_min']}–{a_['gamma_max']}", "photometric"),
            ("Brightness", "brightness", f"±{a_['brightness_limit']}", "photometric"),
            ("Contrast", "contrast", f"±{a_['contrast_limit']}", "photometric"),
            ("Motion blur", "motion_blur", f"k={a_['motion_blur_kernel']}", "photometric"),
            ("Mult. noise", "mult_noise", f"σ={a_['mult_sigma']}", "photometric"),
            ("Gaussian noise", "gaussian", f"σ={a_['gauss_std']}", "photometric"),
            ("Poisson noise", "poisson", f"{a_['poisson_strength']}", "photometric"),
            ("Rotation", "rotation", f"±{a_['rotation_deg']}°", "geometric"),
            ("Translation", "shift", f"±{a_['shift_pct']}%", "geometric"),
            ("Scaling", "scale", f"±{a_['scale_pct']}%", "geometric"),
        ]
        probkey = {"gamma": "gamma_p", "brightness": "brightness_p", "contrast": "contrast_p", "motion_blur": "motion_blur_p",
                   "mult_noise": "mult_p", "gaussian": "gauss_p", "poisson": "poisson_p", "rotation": "rotation_p",
                   "shift": "shift_p", "scale": "scale_p"}
        active = [{"operation": nm, "p": float(a_[probkey[k]]), "range": rng_, "type": typ}
                  for nm, k, rng_, typ in aug_ops if a_[k]]
        a, b = st.columns([1.2, 1])
        with a:
            card_title("Augmentation policy · application probability")
            if active:
                st.altair_chart(alt.Chart(pd.DataFrame(active)).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("operation:N", title=None, sort="-x"),
                    x=alt.X("p:Q", title="Probability", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    color=alt.Color("type:N", title=None, scale=alt.Scale(domain=["photometric", "geometric"], range=["#0B63F6", "#F59E0B"])),
                    tooltip=["operation", "range", alt.Tooltip("p:Q", format=".0%")]
                ).properties(height=40 * len(active) + 30), width="stretch")
                p_none = float(np.prod([1 - x["p"] for x in active]))
                st.caption(f"Expected share of attempts with no operation triggered: {p_none:.1%} "
                           "(these are rejected as exact duplicates by the quality gate).")
            else:
                st.info("No augmentation operation is enabled.")
        with b:
            card_title("Quality gate")
            q = S.qc
            gate = pd.DataFrame([
                {"criterion": "SSIM ≥", "value": q["min_ssim"], "applies to": "photometric"},
                {"criterion": "EPI ≥", "value": q["min_epi"], "applies to": "photometric"},
                {"criterion": "AMBE ≤", "value": q["max_ambe"], "applies to": "photometric"},
                {"criterion": "Content loss ≤", "value": q["max_content_loss"], "applies to": "geometric"},
                {"criterion": "Sibling MAD ≥", "value": q["min_mad_to_siblings"], "applies to": "all"},
                {"criterion": "Exact duplicate", "value": "reject" if q["reject_exact_duplicates"] else "keep", "applies to": "all"},
            ])
            gate["value"] = gate["value"].astype(str)
            st.dataframe(gate, hide_index=True, width="stretch")
            st.caption(("🟢 Gate enabled" if q["enabled"] else "⚪ Gate disabled") +
                       f" · up to {q['max_attempts']} attempts per image · region-aware limits: {'on' if S.region_aware else 'off'}")

        plan = core.balancing_plan(S.records, S.balance_unit, S.target_mode, S.fixed_target)
        if len(plan):
            card_title(f"Balancing plan · {S.balance_unit} · {S.target_mode.lower()}")
            pl = plan.melt(id_vars="cell", value_vars=["count", "to_generate"], var_name="part", value_name="images")
            pl["part"] = pl["part"].map({"count": "original", "to_generate": "to generate"})
            st.altair_chart(alt.Chart(pl).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("cell:N", title=None, sort=alt.SortField("cell"), axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("images:Q", title="Images", stack=True),
                color=alt.Color("part:N", title=None, scale=alt.Scale(domain=["original", "to generate"], range=["#0B63F6", "#A5C8FF"])),
                order=alt.Order("part:N", sort="ascending"),
                tooltip=["cell", "part", "images"]).properties(height=280), width="stretch")

    # ---------------- Last production run ----------------
    with tab_run:
        if S.last_manifest is None:
            st.info("No production run yet. Generate a dataset on the Produce page to see run analytics here.")
        else:
            mdf, rej, s = S.last_manifest, S.last_rejected, S.last_summary
            aug = mdf[mdf["type"] == "augmented"].copy()
            r1 = st.columns(4)
            attempts = s["accepted"] + s["rejected"]
            with r1[0]: metric_card(s["accepted"], "Accepted", f"of {s['requested']} requested")
            with r1[1]: metric_card(s["rejected"], "Rejected attempts", f"{attempts} attempts in total")
            with r1[2]: metric_card(f"{s['accepted'] / max(attempts, 1):.0%}", "Acceptance rate", "Quality gate")
            with r1[3]: metric_card(len(mdf), "Final dataset", "Originals + augmented")
            if aug.empty:
                st.info("The last run produced no augmented images.")
            else:
                a, b = st.columns(2)
                with a:
                    card_title("How often each operation was applied")
                    ops = aug["operations"].fillna("").str.split(";").explode()
                    ops = ops[ops != ""].str.split("=").str[0].value_counts().rename_axis("operation").reset_index(name="images")
                    ops["share"] = ops["images"] / len(aug)
                    st.altair_chart(alt.Chart(ops).mark_bar(cornerRadiusEnd=4, color="#0B63F6").encode(
                        y=alt.Y("operation:N", sort="-x", title=None),
                        x=alt.X("share:Q", title="Share of accepted images", axis=alt.Axis(format="%")),
                        tooltip=["operation", "images", alt.Tooltip("share:Q", format=".0%")]).properties(height=260), width="stretch")
                with b:
                    card_title("Outcome of all attempts")
                    oc = pd.DataFrame({"outcome": ["accepted"], "n": [s["accepted"]]})
                    if rej is not None and not rej.empty:
                        oc = pd.concat([oc, rej["reason"].value_counts().rename_axis("outcome").reset_index(name="n")])
                    st.altair_chart(alt.Chart(oc).mark_arc(innerRadius=70, cornerRadius=4).encode(
                        theta="n:Q",
                        color=alt.Color("outcome:N", title=None, sort=list(oc["outcome"]),
                                        scale=alt.Scale(domain=list(oc["outcome"]),
                                                        range=["#10B981", "#EF4444", "#F59E0B", "#8B5CF6", "#64748B", "#EC4899", "#06B6D4"][:len(oc)])),
                        tooltip=["outcome", "n"]).properties(height=260), width="stretch")

                qm = st.segmented_control("Quality indicator", ["SSIM", "EPI", "AMBE", "Content_loss", "Sibling_MAD"], default="SSIM")
                qm = qm or "SSIM"
                vals = aug[["region", qm]].copy()
                vals[qm] = pd.to_numeric(vals[qm], errors="coerce")
                vals = vals.dropna()
                if vals.empty:
                    st.info(f"{qm} is not available for the accepted images (e.g., only geometric operations were applied).")
                else:
                    a, b = st.columns([1.3, 1])
                    with a:
                        card_title(f"{qm} distribution")
                        st.altair_chart(alt.Chart(vals).mark_bar(color="#0B63F6", opacity=0.85, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                            x=alt.X(f"{qm}:Q", bin=alt.Bin(maxbins=30), title=qm),
                            y=alt.Y("count():Q", title="Images")).properties(height=250), width="stretch")
                    with b:
                        card_title(f"{qm} by region")
                        st.altair_chart(alt.Chart(vals).mark_boxplot(size=22, median={"color": "white"}).encode(
                            x=alt.X("region:N", title=None, axis=alt.Axis(labelAngle=-30)),
                            y=alt.Y(f"{qm}:Q", title=qm, scale=alt.Scale(zero=False)),
                            color=alt.Color("region:N", scale=alt.Scale(range=PALETTE), legend=None)).properties(height=250), width="stretch")

                card_title("Before / after")
                src_name = st.selectbox("Source image", sorted(aug["source"].unique()))
                rec = next((r for r in S.records if r["name"] == src_name), None)
                if rec is not None:
                    kids = aug[aug["source"] == src_name].head(5)
                    zdata = S.get("last_zip")
                    cols = st.columns(len(kids) + 1)
                    cols[0].image(get_pre_cache()[src_name], caption="preprocessed source", width="stretch")
                    if zdata:
                        with zipfile.ZipFile(io.BytesIO(zdata)) as z:
                            for c_, (_, row) in zip(cols[1:], kids.iterrows()):
                                c_.image(z.read(row["output"]), caption=(row["operations"] or "").replace(";", " · "), width="stretch")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
elif page == "Dataset":
    st.markdown("## Dataset")
    a, b = st.columns([1.35, 1])
    imgs = a.file_uploader("X-ray images", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], accept_multiple_files=True)
    meta = b.file_uploader("Metadata table (CSV / Excel, optional)", type=["xlsx", "xls", "csv"])
    if meta is not None:
        md = pd.read_excel(meta) if meta.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(meta)
        S.metadata = md
        st.markdown("### Metadata column mapping")
        opts = [NONE] + list(md.columns)
        def guess(names):
            for i, o in enumerate(opts):
                if str(o).strip().lower() in names:
                    return i
            return 0
        c1, c2, c3, c4 = st.columns(4)
        S.cols["file"] = c1.selectbox("Filename *", list(md.columns), index=max(guess({"filename", "file", "image"}) - 1, 0))
        S.cols["region"] = c2.selectbox("Region", opts, index=guess({"region", "bölge", "bolge"}))
        S.cols["label"] = c3.selectbox("Class label (optional)", opts, index=guess({"class", "label", "sınıf", "sinif"}))
        S.cols["caption"] = c4.selectbox("Caption / report text (optional)", opts, index=guess({"caption", "report", "text"}))
        st.caption("Images without a metadata region are assigned by filename keywords. "
                   "Without a class label, balancing is performed per region only. "
                   "The caption is carried unchanged to every output (preprocessed and augmented) in manifest.csv.")
        st.dataframe(md.head(10), width="stretch")

    if st.button("Load dataset", type="primary"):
        recs, seen_px, seen_name = [], {}, {}
        dup_content, dup_name = [], []
        for f in imgs or []:
            data = f.getvalue()
            px = core.pixel_hash(data)
            if f.name in seen_name:
                (dup_content if seen_name[f.name] == px else dup_name).append(f.name)
                continue
            if px in seen_px:
                dup_content.append(f"{f.name} (= {seen_px[px]})")
                continue
            seen_px[px], seen_name[f.name] = f.name, px
            recs.append({"name": f.name, "data": data, "sha256": core.sha256(data), "pixel_hash": px,
                         "region": core.guess_region(f.name), "label": "unlabeled", "caption": "", "matched": False})
        S.records = recs
        apply_metadata()
        S.pre_cache_key = None
        st.success(f"{len(recs)} unique images loaded.")
        if dup_content:
            st.warning(f"{len(dup_content)} duplicate image(s) skipped (identical pixel content): " + ", ".join(dup_content[:15])
                       + (" …" if len(dup_content) > 15 else ""))
        if dup_name:
            st.error(f"{len(dup_name)} file(s) share a filename with a different image and were skipped: "
                     + ", ".join(dup_name[:15]) + ". Rename them so every filename is unique.")
        if S.metadata is not None and S.cols.get("file"):
            unmatched = [r["name"] for r in recs if not r["matched"]]
            names = set(S.metadata[S.cols["file"]].astype(str).str.strip())
            missing = sorted(names - {r["name"] for r in recs})
            if unmatched:
                st.warning(f"{len(unmatched)} image(s) not found in the metadata table: " + ", ".join(unmatched[:15]))
            if missing:
                st.info(f"{len(missing)} metadata row(s) have no uploaded image: " + ", ".join(missing[:15])
                        + (" …" if len(missing) > 15 else ""))

    if S.records:
        st.markdown("### Records")
        ed = st.data_editor(records_df(), width="stretch", hide_index=True,
                            disabled=["filename"], key="rec_editor")
        if st.button("Save edits"):
            mp = {row.filename: row for row in ed.itertuples()}
            for r in S.records:
                row = mp[r["name"]]
                r["label"], r["region"] = str(row.label), str(row.region)
                r["caption"] = "" if pd.isna(row.caption) else str(row.caption)
            st.success("Saved.")

# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------
elif page == "Preprocess":
    st.markdown("## Preprocess")
    c = S.pre
    with st.expander("Size & normalization", expanded=True):
        a, b, cx = st.columns(3)
        sizes = [224, 384, 512, 640, 768, 896, 1024]
        c["resize"] = a.selectbox("Resize", sizes, index=sizes.index(c["resize"]))
        modes = ["Stretch", "Preserve aspect + padding"]
        c["resize_mode"] = a.radio("Resize behavior", modes, index=modes.index(c["resize_mode"]))
        if c["resize_mode"] != "Stretch":
            c["padding_value"] = a.slider("Padding value", 0, 255, int(c["padding_value"]))
        norms = ["None", "Min-Max", "Percentile"]
        c["normalization"] = b.radio("Normalization", norms, horizontal=True, index=norms.index(c["normalization"]))
        if c["normalization"] == "Min-Max":
            c["minmax_low"] = cx.number_input("Output min", 0, 254, int(c["minmax_low"]))
            c["minmax_high"] = cx.number_input("Output max", 1, 255, int(c["minmax_high"]))
        elif c["normalization"] == "Percentile":
            c["p_low"] = cx.number_input("Low percentile", 0.0, 20.0, float(c["p_low"]), 0.5)
            c["p_high"] = cx.number_input("High percentile", 80.0, 100.0, float(c["p_high"]), 0.5)
    with st.expander("Contrast", expanded=True):
        a, b, cx = st.columns(3)
        c["clahe"] = a.toggle("CLAHE", c["clahe"])
        c["clahe_clip"] = b.slider("CLAHE clip limit", 0.5, 5.0, float(c["clahe_clip"]), 0.1)
        c["clahe_grid"] = cx.select_slider("CLAHE grid", [4, 6, 8, 10, 12, 16], value=int(c["clahe_grid"]))
        c["hist_eq"] = a.toggle("Global histogram equalization", c["hist_eq"])
    with st.expander("Denoising & detail"):
        a, b, cx = st.columns(3)
        c["bilateral"] = a.toggle("Bilateral", c["bilateral"])
        c["bilateral_d"] = a.slider("Bilateral d", 3, 11, int(c["bilateral_d"]), 2)
        c["bilateral_sigma_color"] = b.slider("Sigma color", 10, 120, int(c["bilateral_sigma_color"]), 5)
        c["bilateral_sigma_space"] = cx.slider("Sigma space", 10, 120, int(c["bilateral_sigma_space"]), 5)
        c["median"] = a.toggle("Median", c["median"])
        c["median_ksize"] = b.select_slider("Median kernel", [3, 5, 7], value=int(c["median_ksize"]))
        c["wavelet"] = a.toggle("Wavelet denoising", c["wavelet"], disabled=not core.WAVELET_AVAILABLE)
        fams = ["db1", "db4", "sym4"]
        c["wavelet_family"] = b.selectbox("Wavelet family", fams, index=fams.index(c["wavelet_family"]))
        c["wavelet_level"] = cx.select_slider("Wavelet level", [1, 2, 3], value=int(c["wavelet_level"]))
        c["wavelet_threshold"] = cx.selectbox("Threshold mode", ["soft", "hard"], index=["soft", "hard"].index(c["wavelet_threshold"]))
        c["unsharp"] = a.toggle("Unsharp masking", c["unsharp"])
        c["unsharp_radius"] = b.slider("Unsharp radius", 0.5, 3.0, float(c["unsharp_radius"]), 0.1)
        c["unsharp_percent"] = cx.slider("Unsharp amount (%)", 10, 250, int(c["unsharp_percent"]), 10)
        c["gamma_pre"] = a.toggle("Gamma correction", c["gamma_pre"])
        c["gamma_pre_value"] = b.slider("Gamma", 0.6, 1.5, float(c["gamma_pre_value"]), 0.05)
    with st.expander("Illumination & morphology"):
        a, b = st.columns(2)
        c["homomorphic"] = a.toggle("Homomorphic filter", c["homomorphic"])
        c["homo_gamma_l"] = a.slider("γL", 0.1, 1.0, float(c["homo_gamma_l"]), 0.05)
        c["homo_gamma_h"] = a.slider("γH", 1.0, 3.0, float(c["homo_gamma_h"]), 0.1)
        c["homo_cutoff"] = a.slider("Cutoff D0", 5.0, 100.0, float(c["homo_cutoff"]), 5.0)
        b.caption("Morphological contrast enhancement: I + TopHat(I) − BlackHat(I)")
        c["tophat"] = b.toggle("Add white top-hat (bright details)", c["tophat"])
        c["tophat_kernel"] = b.select_slider("Top-hat kernel", [3, 5, 7, 9, 11, 15, 21], value=int(c["tophat_kernel"]))
        c["blackhat"] = b.toggle("Subtract black-hat (dark details)", c["blackhat"])
        c["blackhat_kernel"] = b.select_slider("Black-hat kernel", [3, 5, 7, 9, 11, 15, 21], value=int(c["blackhat_kernel"]))

    if S.records:
        rec = st.selectbox("Preview image", [r["name"] for r in S.records])
        rr = next(r for r in S.records if r["name"] == rec)
        src = core.pil_from_bytes(rr["data"])
        out = core.preprocess(src, c)
        a, b = st.columns(2)
        a.image(src, caption="Original", width="stretch")
        b.image(out, caption="Preprocessed", width="stretch")
        st.dataframe(pd.DataFrame([core.metrics(src, out)]), width="stretch")

        st.divider()
        st.markdown("### Download preprocessed images")
        st.caption("Applies the current preprocessing settings to all images and exports them without augmentation. "
                   "Regenerate the ZIP after changing the settings.")
        if st.button("Generate preprocessed ZIP", type="primary"):
            cache = get_pre_cache()
            mem, rows = io.BytesIO(), []
            bar = st.progress(0.0)
            with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
                for i, r in enumerate(S.records, 1):
                    src = core.pil_from_bytes(r["data"])
                    pre = cache[r["name"]]
                    data = core.png_bytes(pre)
                    path = core.output_path(r, "pre")
                    z.writestr(path, data)
                    rows.append({"output": path, "source": r["name"], "label": r["label"], "region": r["region"],
                                 "caption": r.get("caption", ""),
                                 **core.metrics(src, pre), "sha256": core.sha256(data), "dhash": hex(core.dhash(pre))})
                    bar.progress(i / len(S.records))
                z.writestr("preprocess_metrics.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"))
                z.writestr("preprocess_config.json", json.dumps({
                    "xpaug_version": core.__version__, "project": S.project_name, "preprocessing": c,
                }, indent=2, ensure_ascii=False).encode("utf-8"))
            S.pre_zip = mem.getvalue()
            st.success(f"{len(S.records)} images preprocessed.")
        if S.get("pre_zip"):
            st.download_button("Download preprocessed ZIP", S.pre_zip, f"{S.project_name}_preprocessed.zip",
                               "application/zip", type="primary")
    else:
        st.info("Load images on the Dataset page first.")

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
elif page == "Augmentation":
    st.markdown("## Augmentation")
    c = S.aug
    with st.expander("Intensity", expanded=True):
        a, b, cx = st.columns(3)
        c["gamma"] = a.toggle("Gamma", c["gamma"])
        c["gamma_min"], c["gamma_max"] = b.slider("Gamma range", 0.60, 1.50, (float(c["gamma_min"]), float(c["gamma_max"])), 0.01)
        c["gamma_p"] = cx.slider("Gamma p", 0.0, 1.0, float(c["gamma_p"]), 0.05)
        c["contrast"] = a.toggle("Contrast", c["contrast"])
        c["contrast_limit"] = b.slider("Contrast ±", 0.0, 0.40, float(c["contrast_limit"]), 0.01)
        c["contrast_p"] = cx.slider("Contrast p", 0.0, 1.0, float(c["contrast_p"]), 0.05)
        c["brightness"] = a.toggle("Brightness", c["brightness"])
        c["brightness_limit"] = b.slider("Brightness ±", 0.0, 0.20, float(c["brightness_limit"]), 0.01)
        c["brightness_p"] = cx.slider("Brightness p", 0.0, 1.0, float(c["brightness_p"]), 0.05)
    with st.expander("Noise & blur", expanded=True):
        a, b, cx = st.columns(3)
        c["mult_noise"] = a.toggle("Multiplicative noise", c["mult_noise"])
        c["mult_sigma"] = b.slider("Multiplicative σ", 0.0, 0.20, float(c["mult_sigma"]), 0.01)
        c["mult_p"] = cx.slider("Multiplicative p", 0.0, 1.0, float(c["mult_p"]), 0.05)
        c["gaussian"] = a.toggle("Gaussian noise", c["gaussian"])
        c["gauss_std"] = b.slider("Gaussian σ", 0.0, 10.0, float(c["gauss_std"]), 0.5)
        c["gauss_p"] = cx.slider("Gaussian p", 0.0, 1.0, float(c["gauss_p"]), 0.05)
        c["poisson"] = a.toggle("Poisson noise", c["poisson"])
        c["poisson_strength"] = b.slider("Poisson strength", 0.05, 0.60, float(c["poisson_strength"]), 0.05)
        c["poisson_p"] = cx.slider("Poisson p", 0.0, 1.0, float(c["poisson_p"]), 0.05)
        c["motion_blur"] = a.toggle("Motion blur", c["motion_blur"])
        c["motion_blur_kernel"] = b.select_slider("Motion blur kernel", [3, 5, 7, 9], value=int(c["motion_blur_kernel"]))
        c["motion_blur_p"] = cx.slider("Motion blur p", 0.0, 1.0, float(c["motion_blur_p"]), 0.05)
    with st.expander("Geometry", expanded=True):
        a, b, cx = st.columns(3)
        c["rotation"] = a.toggle("Rotation", c["rotation"])
        c["rotation_deg"] = b.slider("Rotation ±°", 0.0, 30.0, float(c["rotation_deg"]), 0.5)
        c["rotation_p"] = cx.slider("Rotation p", 0.0, 1.0, float(c["rotation_p"]), 0.05)
        c["shift"] = a.toggle("Translation", c["shift"])
        c["shift_pct"] = b.slider("Translation ±%", 0.0, 8.0, float(c["shift_pct"]), 0.1)
        c["shift_p"] = cx.slider("Translation p", 0.0, 1.0, float(c["shift_p"]), 0.05)
        c["scale"] = a.toggle("Scaling", c["scale"])
        c["scale_pct"] = b.slider("Scaling ±%", 0.0, 12.0, float(c["scale_pct"]), 0.5)
        c["scale_p"] = cx.slider("Scaling p", 0.0, 1.0, float(c["scale_p"]), 0.05)

    S.region_aware = st.toggle("Use region-specific geometric limits", S.region_aware)
    if S.region_aware and S.records:
        st.markdown("### Region overrides")
        rr = st.selectbox("Region", sorted({r["region"] for r in S.records}))
        ov = S.region_overrides.get(rr, {})
        a, b, cx = st.columns(3)
        rot = a.number_input("Rotation limit (±°)", 0.0, 30.0, float(ov.get("rotation_deg", c["rotation_deg"])), 0.5)
        sh = b.number_input("Translation limit (±%)", 0.0, 8.0, float(ov.get("shift_pct", c["shift_pct"])), 0.1)
        sc = cx.number_input("Scaling limit (±%)", 0.0, 12.0, float(ov.get("scale_pct", c["scale_pct"])), 0.5)
        if st.button("Save override"):
            S.region_overrides[rr] = {"rotation_deg": rot, "shift_pct": sh, "scale_pct": sc}
            st.success("Saved.")
        if S.region_overrides:
            st.dataframe(pd.DataFrame(S.region_overrides).T, width="stretch")

    st.markdown("### Quality gate")
    q = S.qc
    q["enabled"] = st.toggle("Reject generated images that fail the quality criteria", q["enabled"])
    a, b, cx, d = st.columns(4)
    q["min_ssim"] = a.number_input("Min SSIM", 0.0, 1.0, float(q["min_ssim"]), 0.05)
    q["min_epi"] = b.number_input("Min EPI", 0.0, 1.0, float(q["min_epi"]), 0.05)
    q["max_ambe"] = cx.number_input("Max AMBE", 0.0, 128.0, float(q["max_ambe"]), 1.0)
    q["min_mad_to_siblings"] = d.number_input("Min difference to siblings (MAD, grey levels)", 0.0, 50.0, float(q.get("min_mad_to_siblings", 1.0)), 0.5)
    q["max_content_loss"] = a.number_input("Max content loss (geometric)", 0.0, 1.0, float(q.get("max_content_loss", 0.02)), 0.005, format="%.3f")
    q["reject_exact_duplicates"] = b.checkbox("Reject exact duplicates of source", q["reject_exact_duplicates"])
    q["max_attempts"] = cx.number_input("Max attempts per image", 1, 20, int(q["max_attempts"]))
    st.caption("Photometric thresholds (SSIM, EPI, AMBE) apply only when a photometric operation was applied; "
               "the content-loss threshold applies only when a geometric operation was applied. "
               "Rejected images are replaced by new attempts with different seeds and logged in rejected.csv.")

    if S.records:
        st.markdown("### Preview")
        rec = st.selectbox("Preview image", [r["name"] for r in S.records], key="augprev")
        rr = next(r for r in S.records if r["name"] == rec)
        pre = get_pre_cache()[rr["name"]]
        k = st.number_input("Preview variant", 1, 999, 1)
        cfg = core.region_cfg(c, S.region_overrides, rr["region"], S.region_aware)
        pmask = core.foreground_mask(pre)
        ar = core.augment(pre, cfg, core.derive_seed(S.seed, "preview", rr["name"], k), pmask)
        out, ops = ar.image, ar.ops
        m = core.augmented_metrics(pre, ar, pmask)
        ok, reason = core.quality_gate(m, q, None)
        a, b = st.columns(2)
        a.image(pre, caption="Preprocessed", width="stretch")
        b.image(out, caption=f"Augmented — {'ACCEPTED' if ok else 'REJECTED: ' + reason}", width="stretch")
        st.caption("Operations: " + (", ".join(ops) if ops else "none triggered"))
        if ar.geometric and not ar.photometric:
            st.info("Only geometric operations were applied: fidelity indicators (SSIM, PSNR, AMBE, LCR, EPI) are not "
                    "informative and are reported as N/A. The output is assessed by content loss and fill fraction.")
        with st.expander("Foreground mask used for content loss"):
            c1, c2 = st.columns(2)
            c1.image(pmask, caption="Source foreground (Otsu)", width="stretch")
            if ar.mask is not None:
                c2.image(ar.mask, caption="After geometric transform", width="stretch")
        st.dataframe(pd.DataFrame([m]), width="stretch")

# ---------------------------------------------------------------------------
# Produce
# ---------------------------------------------------------------------------
elif page == "Produce":
    st.markdown("## Produce")
    if not S.records:
        st.warning("Load a dataset first.")
        st.stop()
    df = records_df()
    units = ["Class x Region", "Region", "Class"]
    S.balance_unit = st.radio("Balancing unit", units, horizontal=True, index=units.index(S.balance_unit))
    if S.balance_unit != "Region" and df["label"].nunique() <= 1:
        st.info("Only one class label is present; class-aware balancing has no effect.")
    modes = ["Match largest cell", "Fixed target", "No balancing"]
    S.target_mode = st.radio("Target", modes, horizontal=True, index=modes.index(S.target_mode))
    if S.target_mode == "Fixed target":
        S.fixed_target = int(st.number_input("Target per cell", 1, value=int(S.fixed_target)))
    S.max_per_source = int(st.number_input(
        "Max augmented images per source image (0 = no limit)", 0, 100, int(S.get("max_per_source", 0)),
        help="Prevents a few source images from dominating a small cell. Cells that cannot reach their "
             "target under this limit are reported as shortfall."))
    plan = core.balancing_plan(S.records, S.balance_unit, S.target_mode, S.fixed_target)
    if len(plan):
        plan["amplification"] = (plan["target"] / plan["count"]).round(1)
        if S.get("max_per_source", 0):
            plan["max_reachable"] = plan["count"] * (1 + S.max_per_source)
    st.dataframe(plan, width="stretch", hide_index=True)
    if len(plan) and (plan["amplification"] > 5).any():
        hi = plan[plan["amplification"] > 5]
        st.warning("High amplification (> 5×) in: " + ", ".join(f"{c} ({a}×)" for c, a in zip(hi.cell, hi.amplification))
                   + ". Few source images will produce many derivatives; consider a fixed target or a per-source limit.")
    a, b, c = st.columns(3)
    a.metric("Current images", int(plan["count"].sum()) if len(plan) else 0)
    b.metric("To generate", int(plan["to_generate"].sum()) if len(plan) else 0)
    c.metric("Final dataset", int(plan["target"].sum()) if len(plan) else 0)

    if st.button("Generate dataset", type="primary"):
        cache = get_pre_cache()
        t_pre = float(S.get("pre_time", 0.0))
        bar = st.progress(0.0)
        t1 = time.perf_counter()
        res = core.generate(S.records, cache, plan, S.balance_unit, S.aug, S.region_overrides,
                            S.region_aware, S.qc, S.seed, progress=bar.progress,
                            max_per_source=S.get("max_per_source", 0))
        t_gen = time.perf_counter() - t1
        runtime = {"preprocessing_s": round(t_pre, 2), "generation_s": round(t_gen, 2),
                   "attempts": len(res.manifest) + len(res.rejected),
                   "s_per_attempt": round(t_gen / max(len(res.manifest) + len(res.rejected), 1), 3)}
        rows = []
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            # preprocessed originals
            for r in S.records:
                data = core.png_bytes(cache[r["name"]])
                path = core.output_path(r, "pre")
                z.writestr(path, data)
                rows.append({"output": path, "type": "original", "source": r["name"],
                             "label": r["label"], "region": r["region"], "caption": r.get("caption", ""), "seed": None,
                             "operations": None, "photometric": None, "geometric": None, **{k: None for k in core.METRIC_COLUMNS},
                             "sha256": core.sha256(data), "dhash": hex(core.dhash(cache[r["name"]]))})
            for path, data in res.outputs:
                z.writestr(path, data)
            manifest = pd.DataFrame(rows + res.manifest)
            rejected = pd.DataFrame(res.rejected)
            z.writestr("manifest.csv", manifest.to_csv(index=False).encode("utf-8-sig"))
            z.writestr("rejected.csv", rejected.to_csv(index=False).encode())
            z.writestr("project_config.json", json.dumps({
                "xpaug_version": core.__version__, "project": S.project_name, "seed": S.seed,
                "preprocessing": S.pre, "augmentation": S.aug, "quality_gate": S.qc,
                "region_aware": S.region_aware, "region_overrides": S.region_overrides,
                "balancing_unit": S.balance_unit, "balancing_mode": S.target_mode,
                "fixed_target": S.fixed_target, "max_per_source": S.get("max_per_source", 0),
                "plan": plan.to_dict(orient="records"), "shortfall": res.shortfall,
                "runtime": runtime, "system": system_info(),
            }, indent=2, ensure_ascii=False).encode("utf-8"))
        bar.progress(1.0, text=f"Done: {len(res.manifest)} of {int(plan['to_generate'].sum())} requested images generated"
                               + (f" ({sum(res.shortfall.values())} not reachable under the per-source limit)" if res.shortfall else ""))
        S.last_manifest, S.last_rejected = manifest, rejected
        S.last_zip = mem.getvalue()
        S.last_summary = {"runtime": runtime, "requested": int(plan["to_generate"].sum()), "accepted": len(res.manifest),
                          "rejected": len(res.rejected), "shortfall": res.shortfall}
        if res.shortfall:
            st.warning("Some cells could not reach their target (per-source limit or attempt limit): "
                       + ", ".join(f"{k}: {v} missing" for k, v in res.shortfall.items())
                       + ". This is expected for cells with very few source images and is recorded in project_config.json.")
        st.success(f"{len(res.manifest)} augmented images accepted, {len(res.rejected)} attempts rejected. "
                   f"Generation took {t_gen:.1f} s ({runtime['s_per_attempt']} s per attempt); "
                   f"preprocessing {t_pre:.1f} s.")
        st.download_button("Download ZIP", mem.getvalue(), f"{S.project_name}_XP-AUG.zip", "application/zip", type="primary")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
elif page == "Results":
    st.markdown("## Results")
    if S.last_manifest is None:
        st.info("No production run yet.")
    else:
        mdf, rej, s = S.last_manifest, S.last_rejected, S.last_summary
        a, b, c, d = st.columns(4)
        a.metric("Requested", s["requested"])
        b.metric("Accepted", s["accepted"])
        c.metric("Rejected attempts", s["rejected"])
        d.metric("Acceptance rate", f"{s['accepted'] / max(s['accepted'] + s['rejected'], 1):.1%}")
        aug = mdf[mdf["type"] == "augmented"]
        if aug.empty:
            st.info("No augmented images were generated (targets not above current counts, or 'No balancing').")
        else:
            cols = ["SSIM", "PSNR_dB", "AMBE", "LCR", "EPI", "EME_aug", "dHash_distance", "Content_loss", "Fill_fraction"]
            st.caption(f"Photometric: {int(aug['photometric'].astype(bool).sum())} · "
                       f"geometric only: {int((~aug['photometric'].astype(bool) & aug['geometric'].astype(bool)).sum())} images. "
                       "Fidelity indicators are summarized over images with a photometric operation only.")
            num = aug[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            st.markdown("### Quality indicators (accepted augmented images)")
            st.dataframe(num.describe().T, width="stretch")
            st.markdown("### Per region (median)")
            st.dataframe(pd.concat([aug[["region"]], num], axis=1).groupby("region").median(), width="stretch")
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.hist(num["SSIM"].dropna(), bins=30)
            ax.set_xlabel("SSIM")
            ax.set_ylabel("Images")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig, width="content")
            plt.close(fig)
        if rej is not None and not rej.empty:
            st.markdown("### Rejection reasons")
            st.dataframe(rej["reason"].value_counts().rename_axis("reason").reset_index(name="count"), width="stretch", hide_index=True)
        st.markdown("### Manifest")
        st.dataframe(mdf, width="stretch", hide_index=True)