"""
XP-AUG revision experiments.

Runs the analyses requested in peer review on an exported XP-AUG run, without
the Streamlit interface. The input is the ZIP produced by the Produce page:
it already contains the preprocessed source images, their class and region
labels (manifest.csv) and the full configuration (project_config.json).

Experiments
-----------
multiseed : the full generation procedure repeated with N global seeds for three
            gate configurations:
              full    - complete quality gate
              legacy  - ablation: gate without the near-duplicate-of-source
                        criterion, sources visited in file-name order
              nogate  - no quality gate (baseline: same policy, nothing rejected)
            Reports acceptance rate (Wilson 95% CI), rejection reasons, shortfall,
            final class ratio, and, for the baseline, how many accepted outputs
            would have violated each criterion.
sweep     : threshold sensitivity. A pool of candidates is generated once with
            the configured policy; every gate threshold is then varied while the
            others stay at their defaults, and the acceptance rate is recomputed.
stress    : gate validation. Each operation is enabled alone (p = 1) at
            increasing magnitudes, far beyond the defaults, to show which
            criterion rejects which kind of degradation.

Usage
-----
    python experiments/run_experiments.py RUN.zip --out results/ --all
    python experiments/run_experiments.py RUN.zip --out results/ --multiseed --seeds 20
    python experiments/run_experiments.py RUN.zip --out results/ --sweep --pool-k 8
    python experiments/run_experiments.py RUN.zip --out results/ --stress --stress-sources 40

"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xpaug_core as core  # noqa: E402

# Categorical palette (fixed order, validated for colour-vision deficiency)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8c6bd6", "#6b6b66"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3dd"

GATE_CRITERIA = {
    # key in qc            metric          direction  sweep grid
    "min_ssim":            ("SSIM",         ">=", np.round(np.arange(0.50, 1.001, 0.02), 3)),
    "min_epi":             ("EPI",          ">=", np.round(np.arange(0.10, 1.001, 0.05), 3)),
    "max_ambe":            ("AMBE",         "<=", np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40])),
    "max_content_loss":    ("Content_loss", "<=", np.round(np.arange(0.0, 0.0501, 0.0025), 4)),
    "min_mad_to_source":   ("Source_MAD",   ">=", np.round(np.arange(0.0, 8.01, 0.5), 2)),
    "min_mad_to_siblings": ("Sibling_MAD",  ">=", np.round(np.arange(0.0, 8.01, 0.5), 2)),
}

STRESS_LEVELS = {
    # operation : (config updates per level, label of the magnitude)
    "gamma":       ([{"gamma_min": 1 - x, "gamma_max": 1 + x} for x in (0.15, 0.3, 0.5, 0.7, 0.9)], "γ range ±"),
    "brightness":  ([{"brightness_limit": x} for x in (0.04, 0.10, 0.20, 0.35, 0.50)], "limit ±"),
    "contrast":    ([{"contrast_limit": x} for x in (0.10, 0.20, 0.40, 0.60, 0.80)], "limit ±"),
    "motion_blur": ([{"motion_blur_kernel": x} for x in (3, 5, 7, 11, 17)], "kernel"),
    "mult_noise":  ([{"mult_sigma": x} for x in (0.05, 0.10, 0.20, 0.35, 0.50)], "σ"),
    "gaussian":    ([{"gauss_std": x} for x in (2, 4, 8, 16, 32)], "σ (grey levels)"),
    "poisson":     ([{"poisson_strength": x} for x in (0.2, 0.5, 1.0, 2.5, 5.0)], "strength"),
    "rotation":    ([{"rotation_deg": x} for x in (3, 8, 15, 25, 45)], "± degrees"),
    "shift":       ([{"shift_pct": x} for x in (1.5, 5, 10, 15, 25)], "± %"),
    "scale":       ([{"scale_pct": x} for x in (2, 5, 10, 20, 30)], "± %"),
}
PROB_KEY = {"gamma": "gamma_p", "brightness": "brightness_p", "contrast": "contrast_p",
            "motion_blur": "motion_blur_p", "mult_noise": "mult_p", "gaussian": "gauss_p",
            "poisson": "poisson_p", "rotation": "rotation_p", "shift": "shift_p", "scale": "scale_p"}


# ---------------------------------------------------------------------------
# Loading an exported run
# ---------------------------------------------------------------------------
def load_run(zip_path: str):
    """Records, preprocessed images and configuration from an exported XP-AUG ZIP."""
    with zipfile.ZipFile(zip_path) as z:
        cfg = json.loads(z.read("project_config.json").decode("utf-8"))
        man = pd.read_csv(io.BytesIO(z.read("manifest.csv")), encoding="utf-8-sig")
        orig = man[man["type"] == "original"]
        records, pre = [], {}
        for _, r in orig.iterrows():
            rec = {"name": str(r["source"]), "label": "" if pd.isna(r["label"]) else str(r["label"]),
                   "region": str(r["region"]), "caption": "" if pd.isna(r.get("caption")) else str(r.get("caption"))}
            records.append(rec)
            pre[rec["name"]] = Image.open(io.BytesIO(z.read(r["output"]))).convert("L")
    return records, pre, cfg


def gate_config(cfg: dict, variant: str) -> dict:
    qc = {**core.DEFAULT_QC, **cfg.get("quality_gate", {})}
    if variant == "full":          # current defaults fill in criteria missing from older exports
        qc["enabled"] = True
    elif variant == "legacy":
        qc["enabled"] = True
        qc["min_mad_to_source"] = 0.0
    elif variant == "nogate":
        qc["enabled"] = False
    else:
        raise ValueError(variant)
    return qc


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def violations(m: dict, qc: dict) -> dict:
    """Which default criteria an (accepted, ungated) output would have failed."""
    nan = core._nan
    return {
        "viol_exact_duplicate": bool(m.get("Exact_duplicate")),
        "viol_near_dup_source": (not nan(m.get("Source_MAD"))) and m["Source_MAD"] < qc["min_mad_to_source"],
        "viol_ssim": (not nan(m.get("SSIM"))) and m["SSIM"] < qc["min_ssim"],
        "viol_epi": (not nan(m.get("EPI"))) and m["EPI"] < qc["min_epi"],
        "viol_ambe": (not nan(m.get("AMBE"))) and m["AMBE"] > qc["max_ambe"],
        "viol_content_loss": (not nan(m.get("Content_loss"))) and m["Content_loss"] > qc["max_content_loss"],
        "viol_near_dup_sibling": (not nan(m.get("Sibling_MAD"))) and m["Sibling_MAD"] < qc["min_mad_to_siblings"],
    }


# ---------------------------------------------------------------------------
# Experiment 1: multiple seeds x gate configurations
# ---------------------------------------------------------------------------
def _one_run(args):
    zip_path, variant, seed = args
    records, pre, cfg = _STATE
    qc = gate_config(cfg, variant)
    plan = core.balancing_plan(records, cfg["balancing_unit"], cfg["balancing_mode"], int(cfg.get("fixed_target", 0)))
    t0 = time.perf_counter()
    res = core.generate(records, pre, plan, cfg["balancing_unit"], cfg["augmentation"],
                        cfg.get("region_overrides", {}), cfg.get("region_aware", False), qc,
                        base_seed=seed, max_per_source=int(cfg.get("max_per_source", 0)),
                        source_order="sorted" if variant == "legacy" else cfg.get("source_order", "sorted"))
    dt = time.perf_counter() - t0
    acc, rej = len(res.manifest), len(res.rejected)
    labels = pd.Series([r["label"] for r in records] + [m["label"] for m in res.manifest]).value_counts()
    row = {"config": variant, "seed": seed, "attempts": acc + rej, "accepted": acc, "rejected": rej,
           "acceptance_rate": acc / max(acc + rej, 1), "shortfall": int(sum(res.shortfall.values())),
           "final_size": len(records) + acc, "seconds": round(dt, 2)}
    for lab, n in labels.items():
        row[f"final_n_{lab}"] = int(n)
    for reason, n in pd.Series([r["reason"] for r in res.rejected], dtype=object).value_counts().items():
        row[f"rej_{reason}"] = int(n)
    vq = gate_config(cfg, "full")
    vio = pd.DataFrame([violations(m, vq) for m in res.manifest]) if res.manifest else pd.DataFrame()
    for c in vio.columns:
        row[f"accepted_{c}"] = int(vio[c].sum())
    if res.manifest:
        mf = pd.DataFrame(res.manifest)
        for col in ("SSIM", "PSNR_dB", "EPI", "AMBE", "Source_MAD", "Sibling_MAD", "Content_loss"):
            s = pd.to_numeric(mf[col], errors="coerce").replace([np.inf], np.nan).dropna()
            if len(s):
                row[f"{col}_median"] = float(s.median())
                row[f"{col}_max"] = float(s.max())
                row[f"{col}_min"] = float(s.min())
    return row


def run_multiseed(zip_path, out: Path, seeds: list[int], configs: list[str], jobs: int):
    tasks = [(zip_path, v, s) for v in configs for s in seeds]
    rows = _map(_one_run, tasks, zip_path, jobs)
    df = pd.DataFrame(rows).sort_values(["config", "seed"])
    df = df.reindex(columns=sorted(df.columns, key=lambda c: (not c in ("config", "seed"), c))).fillna(0)
    df.to_csv(out / "multiseed_runs.csv", index=False)

    summ = []
    for v, g in df.groupby("config", sort=False):
        k, n = int(g["accepted"].sum()), int(g["attempts"].sum())
        lo, hi = wilson(k, n)
        r = {"config": v, "runs": len(g),
             "acceptance_mean": g["acceptance_rate"].mean(), "acceptance_sd": g["acceptance_rate"].std(ddof=1),
             "acceptance_min": g["acceptance_rate"].min(), "acceptance_max": g["acceptance_rate"].max(),
             "pooled_acceptance": k / max(n, 1), "pooled_wilson_lo": lo, "pooled_wilson_hi": hi,
             "accepted_mean": g["accepted"].mean(), "accepted_sd": g["accepted"].std(ddof=1),
             "shortfall_mean": g["shortfall"].mean(), "final_size_mean": g["final_size"].mean()}
        for c in [c for c in df.columns if c.startswith("rej_") or c.startswith("accepted_viol_")]:
            r[c + "_mean"] = g[c].mean()
        summ.append(r)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out / "multiseed_summary.csv", index=False)
    # per-run Wilson CI for each run
    df[["wilson_lo", "wilson_hi"]] = [wilson(int(a), int(n)) for a, n in zip(df["accepted"], df["attempts"])]
    df.to_csv(out / "multiseed_runs.csv", index=False)
    return df, sdf


# ---------------------------------------------------------------------------
# Candidate pools (used by sweep and stress)
# ---------------------------------------------------------------------------
_STATE = None


def _init(zip_path):
    global _STATE
    _STATE = load_run(zip_path)


def _map(fn, tasks, zip_path, jobs):
    if jobs <= 1:
        if _STATE is None:
            _init(zip_path)
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(zip_path,)) as ex:
        return list(ex.map(fn, tasks, chunksize=max(1, len(tasks) // (jobs * 8))))


def _candidates_for_source(args):
    """k ungated candidates for one source; metrics + 128 px thumbnail for sibling checks."""
    name, aug_cfg, k, tag, base_seed = args
    records, pre, cfg = _STATE
    rec = next(r for r in records if r["name"] == name)
    img = pre[name]
    mask = core.foreground_mask(img)
    if aug_cfg is None:
        aug_cfg = core.region_cfg(cfg["augmentation"], cfg.get("region_overrides", {}), rec["region"],
                                  cfg.get("region_aware", False))
    rows = []
    for i in range(1, k + 1):
        seed = core.derive_seed(base_seed, tag, name, i)
        ar = core.augment(img, aug_cfg, seed, mask)
        m = core.augmented_metrics(img, ar, mask)
        rows.append({"source": name, "label": rec["label"], "region": rec["region"], "idx": i, "seed": seed,
                     "operations": ";".join(ar.ops), "photometric": ar.photometric, "geometric": ar.geometric,
                     **m, "_thumb": core._thumb(ar.image).astype(np.uint8)})
    return rows


def evaluate_pool(pool: pd.DataFrame, qc: dict) -> pd.Series:
    """Gate decision for every candidate, processing each source's candidates in
    order so that Sibling MAD is computed against earlier *accepted* siblings,
    exactly as during generation. Returns the rejection reason ('' = accepted)."""
    reasons = pd.Series("", index=pool.index, dtype=object)
    for _, g in pool.groupby("source", sort=False):
        sibs = []
        for idx, row in g.iterrows():
            t = row["_thumb"].astype(np.float32)
            sib = float(min(np.abs(t - s).mean() for s in sibs)) if sibs else None
            ok, reason = core.quality_gate(row, qc, sib)
            if ok:
                sibs.append(t)
            else:
                reasons[idx] = reason
    return reasons


# ---------------------------------------------------------------------------
# Experiment 2: threshold sensitivity
# ---------------------------------------------------------------------------
def run_sweep(zip_path, out: Path, k: int, seed: int, jobs: int):
    records, pre, cfg = _STATE or load_run(zip_path)
    tasks = [(r["name"], None, k, "sweep", seed) for r in records]
    pool = pd.DataFrame([row for rows in _map(_candidates_for_source, tasks, zip_path, jobs) for row in rows])
    base = gate_config(cfg, "full")
    pool.drop(columns="_thumb").to_csv(out / "sweep_candidate_pool.csv", index=False)

    rows = []
    for key, (metric, direction, grid) in GATE_CRITERIA.items():
        for thr in grid:
            qc = dict(base, **{key: float(thr)})
            reasons = evaluate_pool(pool, qc)
            n = len(pool)
            acc = int((reasons == "").sum())
            own = reasons.str.startswith(_reason_prefix(key)).sum()
            lo, hi = wilson(acc, n)
            rows.append({"criterion": key, "metric": metric, "threshold": float(thr), "default": float(base[key]),
                         "candidates": n, "accepted": acc, "acceptance_rate": acc / n,
                         "wilson_lo": lo, "wilson_hi": hi,
                         "rejected_by_this_criterion": int(own), "share_rejected_by_this_criterion": own / n})
    sw = pd.DataFrame(rows)
    sw.to_csv(out / "sweep_thresholds.csv", index=False)
    plot_sweep(sw, out / "fig_sweep_thresholds.png")
    return pool, sw


def _reason_prefix(key: str) -> str:
    return {"min_ssim": "SSIM<", "min_epi": "EPI<", "max_ambe": "AMBE>", "max_content_loss": "Content_loss>",
            "min_mad_to_source": "near_duplicate_of_source", "min_mad_to_siblings": "near_duplicate_of_sibling"}[key]


# ---------------------------------------------------------------------------
# Experiment 3: stress test of the gate
# ---------------------------------------------------------------------------
def _isolated_cfg(base_aug: dict, op: str, upd: dict) -> dict:
    c = copy.deepcopy(base_aug)
    for o in STRESS_LEVELS:          # everything off ...
        c[o] = False
    c[op] = True                     # ... except the operation under test, always applied
    c[PROB_KEY[op]] = 1.0
    c.update(upd)
    return c


def run_stress(zip_path, out: Path, n_sources: int, k: int, seed: int, jobs: int):
    records, pre, cfg = _STATE or load_run(zip_path)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice([r["name"] for r in records], size=min(n_sources, len(records)), replace=False))
    tasks, meta = [], []
    for op, (levels, unit) in STRESS_LEVELS.items():
        for li, upd in enumerate(levels, 1):
            c = _isolated_cfg(cfg["augmentation"], op, upd)
            for name in chosen:
                tasks.append((name, c, k, f"stress|{op}|{li}", seed))
                meta.append((op, li, _level_label(op, upd), unit))
    results = _map(_candidates_for_source, tasks, zip_path, jobs)
    qc = gate_config(cfg, "full")
    rows = []
    for (op, li, lab, unit), cands in zip(meta, results):
        for cnd in cands:
            ok, reason = core.quality_gate(cnd, qc, None)   # siblings excluded: isolates the fidelity criteria
            rows.append({"operation": op, "level": li, "magnitude": lab, "unit": unit, "source": cnd["source"],
                         "seed": cnd["seed"], "operations": cnd["operations"], "accepted": ok,
                         "reason": reason or "accepted",
                         **{m: cnd[m] for m in ("SSIM", "PSNR_dB", "EPI", "AMBE", "Content_loss", "Fill_fraction",
                                                "Source_MAD", "dHash_distance")}})
    st = pd.DataFrame(rows)
    st.to_csv(out / "stress_candidates.csv", index=False)
    summ = (st.groupby(["operation", "level", "magnitude", "unit", "reason"]).size()
              .unstack("reason", fill_value=0).reset_index())
    rc = [c for c in summ.columns if c not in ("operation", "level", "magnitude", "unit")]
    summ["n"] = summ[rc].sum(axis=1)
    summ["rejection_rate"] = 1 - summ.get("accepted", 0) / summ["n"]
    order = {o: i for i, o in enumerate(STRESS_LEVELS)}
    summ = summ.sort_values(["operation", "level"], key=lambda s: s.map(order) if s.name == "operation" else s)
    summ.to_csv(out / "stress_summary.csv", index=False)
    plot_stress(st, out / "fig_stress_test.png")
    return st, summ


def _level_label(op, upd):
    if op == "gamma":
        return f"{upd['gamma_max'] - 1:.2f}"
    return f"{list(upd.values())[0]:g}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_sweep(sw: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(GATE_CRITERIA)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True)
    for ax, key in zip(axes.ravel(), keys):
        g = sw[sw.criterion == key]
        ax.fill_between(g.threshold, g.wilson_lo * 100, g.wilson_hi * 100, color=PALETTE[0], alpha=0.15, lw=0)
        ax.plot(g.threshold, g.acceptance_rate * 100, color=PALETTE[0], lw=2)
        d = g["default"].iloc[0]
        ax.axvline(d, color=MUTED, lw=1, ls="--")
        y = float(np.interp(d, g.threshold, g.acceptance_rate * 100))
        ax.plot([d], [y], "o", ms=5, color=PALETTE[0], mec="white", mew=1.5)
        ax.annotate(f"default {d:g}\n{y:.1f}%", (d, y), xytext=(6, -22), textcoords="offset points",
                    fontsize=7, color=INK)
        name = GATE_CRITERIA[key][0].replace("_", " ")
        ax.set_title(f"{name} {'≥' if GATE_CRITERIA[key][1] == '>=' else '≤'} threshold", fontsize=9, color=INK,
                     loc="left")
        ax.set_ylim(0, 102)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accepted candidates (%)", fontsize=8, color=INK)
    fig.suptitle("Acceptance rate as each threshold varies (other criteria at default; shaded: Wilson 95% CI)",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


REASON_ORDER = ["accepted", "SSIM", "EPI", "AMBE", "Content_loss", "near_duplicate_of_source", "exact_duplicate_of_source"]
REASON_LABEL = {"accepted": "Accepted", "SSIM": "SSIM", "EPI": "EPI", "AMBE": "AMBE", "Content_loss": "Content loss",
                "near_duplicate_of_source": "Near-duplicate of source", "exact_duplicate_of_source": "Exact duplicate"}


def _reason_group(r: str) -> str:
    for k in REASON_ORDER:
        if r.startswith(k):
            return k
    return r


def plot_stress(st: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = st.assign(rg=st.reason.map(_reason_group))
    colors = {"accepted": "#d9d8d2", "SSIM": PALETTE[0], "EPI": PALETTE[1], "AMBE": PALETTE[2],
              "Content_loss": PALETTE[3], "near_duplicate_of_source": PALETTE[4], "exact_duplicate_of_source": PALETTE[5]}
    ops = list(STRESS_LEVELS)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4), sharey=True)
    present = set(st.rg)
    for ax, op in zip(axes.ravel(), ops):
        g = st[st.operation == op]
        tab = (g.groupby(["level", "magnitude", "rg"]).size().unstack("rg", fill_value=0))
        tab = tab.div(tab.sum(axis=1), axis=0) * 100
        x = np.arange(len(tab))
        bottom = np.zeros(len(tab))
        for rg in REASON_ORDER:
            if rg in tab:
                ax.bar(x, tab[rg].values, bottom=bottom, color=colors[rg], width=0.72,
                       edgecolor="white", linewidth=1.0)
                bottom += tab[rg].values
        ax.set_xticks(x, [m for _, m in tab.index], fontsize=7)
        ax.set_xlabel(g.unit.iloc[0], fontsize=7.5, color=MUTED)
        ax.set_title(op.replace("_", " "), fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, 100)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Candidates (%)", fontsize=8, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[r]) for r in REASON_ORDER if r in present]
    fig.legend(handles, [REASON_LABEL[r] for r in REASON_ORDER if r in present], loc="lower center",
               ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle("Gate decisions when each operation is applied alone at increasing magnitude",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_multiseed(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configs = [c for c in ("full", "legacy", "nogate") if c in set(df.config)]
    names = {"full": "Full gate", "legacy": "Without source\nnear-copy criterion", "nogate": "No gate\n(baseline)"}
    viol = [c for c in df.columns if c.startswith("accepted_viol_")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for i, c in enumerate(configs):
        v = df[df.config == c].acceptance_rate * 100
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
        ax.plot(np.full(len(v), i) + jitter, v, "o", ms=5, color=PALETTE[i], mec="white", mew=1, alpha=0.9)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, lw=1.5)
    ax.set_xticks(range(len(configs)), [names[c] for c in configs], fontsize=8)
    ax.set_ylabel("Acceptance rate per run (%)", fontsize=8, color=INK)
    ax.set_title("Acceptance across seeds (bar = mean)", fontsize=9, color=INK, loc="left")
    _style(ax)
    ax = axes[1]
    labels = {"accepted_viol_exact_duplicate": "Exact duplicate", "accepted_viol_near_dup_source": "Near-dup. of source",
              "accepted_viol_ssim": "SSIM", "accepted_viol_epi": "EPI", "accepted_viol_ambe": "AMBE",
              "accepted_viol_content_loss": "Content loss", "accepted_viol_near_dup_sibling": "Near-dup. of sibling"}
    viol = [v for v in labels if v in df.columns]
    w = 0.8 / len(configs)
    for i, c in enumerate(configs):
        g = df[df.config == c]
        share = [(g[v] / g.accepted.clip(lower=1)).mean() * 100 for v in viol]
        ax.barh(np.arange(len(viol)) + (i - (len(configs) - 1) / 2) * w, share, height=w * 0.9,
                color=PALETTE[i], label=names[c].replace("\n", " "))
    ax.set_yticks(range(len(viol)), [labels[v] for v in viol], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Accepted outputs failing a criterion of the full gate (% of accepted, mean over seeds)", fontsize=7.5,
                  color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_zip", help="ZIP exported from the XP-AUG Produce page")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--all", action="store_true", help="run every experiment")
    p.add_argument("--multiseed", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--stress", action="store_true")
    p.add_argument("--seeds", type=int, default=20, help="number of global seeds for --multiseed")
    p.add_argument("--first-seed", type=int, default=None, help="first seed (default: seed of the exported run)")
    p.add_argument("--configs", default="full,legacy,nogate", help="gate configurations for --multiseed")
    p.add_argument("--pool-k", type=int, default=8, help="candidates per source for --sweep")
    p.add_argument("--stress-sources", type=int, default=40, help="sources sampled for --stress")
    p.add_argument("--stress-k", type=int, default=2, help="candidates per source and level for --stress")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records, pre, cfg = load_run(a.run_zip)
    global _STATE
    _STATE = (records, pre, cfg)
    seed0 = int(cfg.get("seed", 42)) if a.first_seed is None else a.first_seed
    print(f"Loaded {len(records)} preprocessed sources from {a.run_zip} "
          f"(XP-AUG {cfg.get('xpaug_version', '?')} export; core {core.__version__}); seed {seed0}")

    log = {"input": str(a.run_zip), "core_version": core.__version__, "args": vars(a),
           "quality_gate_full": gate_config(cfg, "full")}
    t0 = time.perf_counter()
    if a.all or a.multiseed:
        seeds = [seed0 + i for i in range(a.seeds)]
        df, sdf = run_multiseed(a.run_zip, out, seeds, a.configs.split(","), a.jobs)
        plot_multiseed(df, out / "fig_multiseed.png")
        print("\nMulti-seed summary\n" + sdf[["config", "runs", "acceptance_mean", "acceptance_sd",
                                              "pooled_wilson_lo", "pooled_wilson_hi", "shortfall_mean"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if a.all or a.sweep:
        _, sw = run_sweep(a.run_zip, out, a.pool_k, seed0, a.jobs)
        d = sw[np.isclose(sw.threshold, sw["default"])]
        print("\nAcceptance at default thresholds (sweep pool)\n" +
              d[["criterion", "threshold", "acceptance_rate", "share_rejected_by_this_criterion"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if a.all or a.stress:
        _, ss = run_stress(a.run_zip, out, a.stress_sources, a.stress_k, seed0, a.jobs)
        print("\nStress test: rejection rate per operation and level\n" +
              ss.pivot(index="operation", columns="level", values="rejection_rate")
              .reindex(list(STRESS_LEVELS)).to_string(float_format=lambda v: f"{v:.2f}"))
    log["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "experiment_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"\nDone in {log['seconds']} s. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
