# 🩻 XP-AUG

**X-ray Preprocessing and Augmentation with Built-in Quality Control**: a Streamlit application for preparing, balancing and quality-checking augmented radiograph datasets.

XP-AUG takes a set of X-ray images (plus an optional metadata table) and produces a balanced, reproducible training dataset. It applies a deterministic preprocessing chain and a probability-controlled augmentation policy. Each generated image passes a quantitative quality gate before it is accepted, and every output is traced back to its source image, random seed and applied operations.

---

## Features

- **Dataset profiling**: class × region heatmap, acquisition geometry, brightness/contrast scatter, entropy and dynamic-range distributions, per-region intensity histograms.
- **Deterministic preprocessing**: resize (stretch or aspect-preserving padding), bilateral / median / wavelet denoising, homomorphic filtering, global histogram equalization, morphological contrast enhancement (I + top-hat − black-hat), CLAHE, min-max or percentile normalization, unsharp masking, gamma correction.
- **Stochastic augmentation**: every operation has its own range and application probability.
  - *Photometric*: gamma, brightness, contrast, motion blur, multiplicative / Gaussian / Poisson noise
  - *Geometric*: rotation, translation, scaling (optional region-specific limits)
- **Quality gate**: rejects a generated image and retries with a new seed when it fails any of the criteria below.

  | Criterion | Applies to | Default |
  |---|---|---|
  | SSIM ≥ | photometric operations | 0.60 |
  | EPI (edge preservation) ≥ | photometric operations | 0.30 |
  | AMBE (mean brightness shift) ≤ | photometric operations | 25 |
  | Content loss ≤ | geometric operations | 0.02 |
  | MAD to sibling outputs ≥ | all | 1.0 |
  | Exact duplicate of source | all | rejected |

  Fidelity indicators are computed against the source warped by the *same* geometric transform. They therefore measure photometric degradation only, not spatial displacement. For purely geometric outputs they are reported as N/A.
- **Balancing**: per *Class × Region*, *Class* or *Region*, either to match the largest cell or to reach a fixed target. An optional per-source cap prevents a few images from dominating a small cell.
- **Reproducibility**: each output gets a seed derived from the global seed, balancing cell, source file and attempt index. The same configuration always produces byte-identical results.
- **Quality indicators**: SSIM, PSNR, AMBE, LCR, EPI, entropy, EME, dHash distance, content loss, fill fraction, SHA-256.

## Workflow

```
Overview → Dataset → Preprocess → Augmentation → Produce → Results
```

| Page | What happens |
|---|---|
| **Overview** | Dataset profile, image statistics, active pipeline, analytics of the last production run |
| **Dataset** | Upload images and an optional metadata table. Duplicate detection by pixel content. Inline editing of labels and regions |
| **Preprocess** | Configure the preprocessing chain, preview it, export preprocessed images as a ZIP |
| **Augmentation** | Configure the augmentation policy, region overrides and quality gate. Live preview with accept/reject decision |
| **Produce** | Review the balancing plan and generate the final dataset as a ZIP |
| **Results** | Summary statistics of quality indicators, rejection reasons, full manifest |

## Installation

Requires **Python 3.12 or newer**.

```bash
git clone https://github.com/<user>/XP-AUG.git
cd XP-AUG

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Metadata table (optional)

A CSV or Excel file with one row per image. The column names are mapped in the UI. Common names are detected automatically.

| Column | Required | Auto-detected names | Purpose |
|---|---|---|---|
| Filename | yes | `filename`, `file`, `image` | Must match the uploaded file name exactly |
| Region | no | `region`, `bölge`, `bolge` | Anatomical region. If missing, it is guessed from filename keywords |
| Class label | no | `class`, `label`, `sınıf`, `sinif` | Diagnostic group. Without it, balancing runs per region |
| Caption | no | `caption`, `report`, `text` | Free text passed unchanged to every derived image |

Example (Excel or CSV):

| filename | caption | region | class |
|---|---|---|---|
| MPS4A_Image 1.png | There are dysplastic capital femoral epiphyses (CFE) with normally formed iliac wings. Antero-posterior (AP) pelvis and hips demonstrates hypoplastic dense capital femoral epiphyses (CFE), with irregular femoral head contours. | Pelvis/Hip | MPS_IVA |
| MPS6_Image 11.png | The chest X-ray demonstrates paddle-shaped widened ribs and thick irregular clavicles. | Chest | MPS_VI |
| MPS4A_Image 100.png | AP view of the right hand demonstrates proximal pointing of the metacarpal bones, short phalanges and metacarpal bones, hypoplastic carpal bones with delayed ossification, oblique distal radius | Arm | MPS_IVA |

Region names such as `Pelvis/Hip` are converted to folder-safe names (`Pelvis_Hip`) in the output.

### Output

The **Produce** page creates a ZIP file with this layout:

```
<label>/<region>/<image>_pre.png        preprocessed originals
<label>/<region>/<image>_aug_001.png    accepted augmented images
manifest.csv                            one row per output: source, seed, operations, quality indicators, hashes
rejected.csv                            every rejected attempt and the reason
project_config.json                     full configuration, balancing plan, shortfall, runtime and system info
```

To reproduce a run, use the same input images and the settings in `project_config.json`.

## Using the core library without the UI

All image processing lives in `xpaug_core.py`, which has no Streamlit dependency:

```python
from PIL import Image
import xpaug_core as core

img = Image.open("example.png").convert("L")
pre = core.preprocess(img, core.DEFAULT_PRE)

mask = core.foreground_mask(pre)
res = core.augment(pre, core.DEFAULT_AUG, seed=42, mask=mask)
m = core.augmented_metrics(pre, res, mask)
ok, reason = core.quality_gate(m, core.DEFAULT_QC, None)

print(res.ops, ok, reason, m["SSIM"])
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use synthetic images only. They cover metric correctness (including SSIM against the `scikit-image` reference implementation), reproducibility, balancing, the quality gate, geometric content loss and per-source limits.

## Project structure

```
XP-AUG/
├── app.py                 Streamlit user interface
├── xpaug_core.py          Preprocessing, augmentation, metrics, balancing (UI-independent)
├── assets/
│   └── idea_logo.png      Sidebar logo
├── tests/
│   └── test_core.py       Unit tests
├── requirements.txt       Runtime dependencies
└── requirements-dev.txt   Test dependencies
```

## Data privacy

This repository contains **no patient images or clinical metadata**. XP-AUG processes all data locally in your session and uploads nothing to external services. Do not commit medical images to this repository. The included `.gitignore` excludes common image, spreadsheet and output formats outside `assets/`.

## Citation

If you use XP-AUG in your research, please cite the accompanying paper (citation details will be added after publication).
