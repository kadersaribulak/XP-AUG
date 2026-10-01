import numpy as np
import pandas as pd
import pytest
from PIL import Image
from skimage.metrics import structural_similarity

import xpaug_core as core


def synthetic_xray(seed=0, size=256):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:size, :size]
    img = 40 + 150 * np.exp(-((x - size / 2) ** 2) / (2 * (size / 8) ** 2))  # "bone" band
    img += 30 * np.sin(y / 9.0) * (img > 100)                                  # trabecular texture
    img += rng.normal(0, 4, img.shape)
    return Image.fromarray(np.uint8(np.clip(img, 0, 255)))


def test_identity_metrics():
    im = synthetic_xray()
    m = core.metrics(im, im)
    assert m["SSIM"] == pytest.approx(1.0)
    assert m["PSNR_dB"] == float("inf")
    assert m["AMBE"] == 0
    assert m["dHash_distance"] == 0
    assert m["Exact_duplicate"] is True
    assert m["LCR"] == pytest.approx(1.0, abs=1e-6)
    assert m["EPI"] == pytest.approx(1.0)


def test_ssim_matches_reference_implementation():
    a = synthetic_xray(0)
    b = Image.fromarray(np.uint8(np.clip(np.asarray(a, float) * 0.9 + 10 + np.random.default_rng(1).normal(0, 6, (256, 256)), 0, 255)))
    ours = core.ssim(a, b)
    ref = structural_similarity(np.asarray(a), np.asarray(b), data_range=255, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False)
    assert ours == pytest.approx(ref, abs=0.02)


def test_augment_is_reproducible_and_seed_sensitive():
    im = synthetic_xray()
    cfg = dict(core.DEFAULT_AUG)
    r1 = core.augment(im, cfg, 123); a1, ops1 = r1.image, r1.ops
    r2 = core.augment(im, cfg, 123); a2, ops2 = r2.image, r2.ops
    assert np.array_equal(np.asarray(a1), np.asarray(a2)) and ops1 == ops2
    outs = {core.augment(im, cfg, s).image.tobytes() for s in range(20)}
    assert len(outs) > 1


def test_derive_seed_differs_across_regions():
    assert core.derive_seed(42, "MPSIVA|hand", "a.png", 1) != core.derive_seed(42, "MPSIVA|spine", "a.png", 1)
    assert core.derive_seed(42, "x", "a.png", 1) == core.derive_seed(42, "x", "a.png", 1)


def test_morphological_enhancement_preserves_anatomy():
    im = np.asarray(synthetic_xray())
    out = core.morphological_enhance(im, True, 11, True, 11)
    # the enhanced image must stay close to the original, unlike a raw top-hat residual
    assert np.corrcoef(im.ravel(), out.ravel())[0, 1] > 0.9
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    residual = cv2.morphologyEx(im, cv2.MORPH_TOPHAT, k)
    assert np.corrcoef(im.ravel(), out.ravel())[0, 1] > np.corrcoef(im.ravel(), residual.ravel())[0, 1]


def make_records(n_images=60):
    regions = ["hand", "spine", "pelvis_hip"]
    return [{"name": f"img{i:03d}.png", "label": "MPS_IVA" if i % 3 else "MPS_VI",
             "region": regions[(i * 7) % 3 if i % 4 else 0]} for i in range(n_images)]


def test_balancing_uses_class_region_cells():
    recs = make_records()
    plan = core.balancing_plan(recs, "Class x Region", "Match largest cell")
    assert plan["count"].sum() == len(recs)
    assert (plan["target"] == plan["count"].max()).all()
    assert all("|" in c for c in plan["cell"])
    region_plan = core.balancing_plan(recs, "Region", "Match largest cell")
    assert set(region_plan["cell"]) == {"hand", "spine", "pelvis_hip"}


def test_generate_respects_plan_and_quality_gate():
    recs = make_records(18)
    cache = {r["name"]: synthetic_xray(i, 128) for i, r in enumerate(recs)}
    plan = core.balancing_plan(recs, "Class x Region", "Fixed target", 6)
    qc = dict(core.DEFAULT_QC)
    res = core.generate(recs, cache, plan, "Class x Region", core.DEFAULT_AUG, {}, False, qc, 42)
    produced = pd.DataFrame(res.manifest)
    assert len(produced) + sum(res.shortfall.values()) == plan["to_generate"].sum()
    assert (produced["SSIM"].dropna() >= qc["min_ssim"]).all()
    assert produced.loc[~produced["photometric"], "SSIM"].isna().all()
    assert (produced["Content_loss"] <= qc["max_content_loss"]).all()
    assert not produced["Exact_duplicate"].any()
    # reproducible
    res2 = core.generate(recs, cache, plan, "Class x Region", core.DEFAULT_AUG, {}, False, qc, 42)
    assert [r["sha256"] for r in res.manifest] == [r["sha256"] for r in res2.manifest]


def test_quality_gate_rejects_duplicates():
    m = core.metrics(synthetic_xray(), synthetic_xray())
    ok, reason = core.quality_gate(m, core.DEFAULT_QC, None)
    assert not ok and reason == "exact_duplicate_of_source"


def _only(**kw):
    cfg = {k: False if isinstance(v, bool) else v for k, v in core.DEFAULT_AUG.items()}
    cfg.update(kw)
    return cfg


def test_rotation_only_reports_fidelity_as_not_applicable():
    im = synthetic_xray()
    mask = core.foreground_mask(im)
    ar = core.augment(im, _only(rotation=True, rotation_deg=8.0, rotation_p=1.0), 5, mask)
    assert ar.geometric and not ar.photometric
    m = core.augmented_metrics(im, ar, mask)
    for k in core.FIDELITY_COLUMNS:
        assert np.isnan(m[k])
    assert m["Exact_duplicate"] is False
    assert 0.0 <= m["Content_loss"] < 0.2 and m["Fill_fraction"] > 0
    ok, _ = core.quality_gate(m, dict(core.DEFAULT_QC, max_content_loss=1.0), None)
    assert ok


def test_photometric_fidelity_is_measured_after_alignment():
    im = synthetic_xray()
    mask = core.foreground_mask(im)
    cfg = _only(rotation=True, rotation_deg=8.0, rotation_p=1.0, gamma=True, gamma_min=1.1, gamma_max=1.1, gamma_p=1.0)
    ar = core.augment(im, cfg, 5, mask)
    m = core.augmented_metrics(im, ar, mask)
    naive = core.metrics(im, ar.image)
    assert ar.photometric and ar.geometric
    assert 0.8 < m["SSIM"] < 1.0 and naive["SSIM"] < m["SSIM"]


def test_content_loss_detects_anatomy_pushed_out_of_frame():
    size = 200
    a = np.zeros((size, size), np.uint8); a[:, 150:195] = 200   # anatomy near the right border
    im = Image.fromarray(a); mask = core.foreground_mask(im)
    ar = core.augment(im, _only(shift=True, shift_pct=8.0, shift_p=1.0), 3, mask)
    m = core.augmented_metrics(im, ar, mask)
    tx = int(ar.ops[0].split("(")[1].split(",")[0])
    if tx > 5:
        assert m["Content_loss"] > 0.02
        ok, reason = core.quality_gate(m, core.DEFAULT_QC, None)
        assert not ok and reason.startswith("Content_loss")
    ar0 = core.augment(im, _only(rotation=True, rotation_deg=0.5, rotation_p=1.0), 3, mask)
    assert core.augmented_metrics(im, ar0, mask)["Content_loss"] < 0.02


def test_photometric_siblings_are_not_treated_as_duplicates():
    """dHash is invariant to gamma; sibling redundancy must be pixel-based."""
    im = synthetic_xray()
    a = core.augment(im, _only(gamma=True, gamma_min=0.9, gamma_max=0.9, gamma_p=1.0), 1).image
    b = core.augment(im, _only(gamma=True, gamma_min=1.1, gamma_max=1.1, gamma_p=1.0), 1).image
    assert core.hamming(core.dhash(a), core.dhash(b)) <= 2          # dHash cannot tell them apart
    assert core.min_mad(b, [core._thumb(a)]) > core.DEFAULT_QC["min_mad_to_siblings"]
    assert core.min_mad(a, [core._thumb(a)]) == 0.0


def test_safe_output_paths_and_caption_inheritance():
    assert core.safe_name("Pelvis/Hip") == "Pelvis_Hip"
    rec = {"name": "Image_201.png", "label": "MPS_IVA", "region": "Pelvis/Hip", "caption": "dysplastic CFE"}
    assert core.output_path(rec, "pre") == "MPS_IVA/Pelvis_Hip/Image_201_pre.png"
    recs = [dict(rec, name=f"Image_{i}.png", caption=f"caption {i}") for i in range(3)]
    recs.append({"name": "Image_9.png", "label": "MPS_VI", "region": "Chest", "caption": "paddle ribs"})
    cache = {r["name"]: synthetic_xray(i, 128) for i, r in enumerate(recs)}
    plan = core.balancing_plan(recs, "Class x Region", "Match largest cell")
    res = core.generate(recs, cache, plan, "Class x Region", core.DEFAULT_AUG, {}, False, core.DEFAULT_QC, 1)
    df = pd.DataFrame(res.manifest)
    assert len(df) == 2 and (df["caption"] == "paddle ribs").all()
    assert all("/" not in p.split("/", 1)[1].rsplit("/", 1)[0] for p in df["output"])


def test_per_source_cap_limits_amplification():
    recs = [{"name": f"f{i}.png", "label": "A", "region": "foot" if i < 2 else "hand", "caption": ""} for i in range(12)]
    cache = {r["name"]: synthetic_xray(i, 128) for i, r in enumerate(recs)}
    plan = core.balancing_plan(recs, "Region", "Match largest cell")          # foot: 2 -> 10
    res = core.generate(recs, cache, plan, "Region", core.DEFAULT_AUG, {}, False, core.DEFAULT_QC, 3, max_per_source=3)
    df = pd.DataFrame(res.manifest)
    assert df.groupby("source").size().max() <= 3
    assert res.shortfall.get("foot") == 8 - 6


def test_pixel_hash_ignores_file_encoding():
    im = synthetic_xray()
    import io
    b1 = core.png_bytes(im)
    buf = io.BytesIO(); im.save(buf, "PNG", compress_level=1); b2 = buf.getvalue()
    assert b1 != b2 and core.pixel_hash(b1) == core.pixel_hash(b2)


def test_region_override_limits_rotation():
    cfg = core.region_cfg(core.DEFAULT_AUG, {"Chest": {"rotation_deg": 3.0}}, "Chest", True)
    im = synthetic_xray()
    for s in range(40):
        ar = core.augment(im, dict(cfg, rotation_p=1.0), s)
        rot = [float(o.split("=")[1]) for o in ar.ops if o.startswith("rotation")]
        assert all(abs(x) <= 3.0 for x in rot)
    assert core.region_cfg(core.DEFAULT_AUG, {"Chest": {"rotation_deg": 3.0}}, "Chest", False)["rotation_deg"] == 8.0
