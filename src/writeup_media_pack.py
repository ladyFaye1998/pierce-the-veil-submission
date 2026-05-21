"""
Curate a media-gallery pack for the Kaggle writeup.

Copies the six highest-impact figures into assets/writeup_media/ with
ordered filenames so the upload order in Kaggle is intuitive.

Author: Lady Faye
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
DST = ROOT / "assets" / "writeup_media"

ORDER = [
    ("00_pipeline.png",                 "01_pipeline_overview.png",
     "End-to-end attack pipeline: encoder identification -> impossibility argument -> calibrated reconstruction -> compliance."),
    ("01_distribution.png",             "02_marginal_distribution.png",
     "Marginal of Z: skew = +0.575, excess kurtosis = +0.894; consistent with a class-imbalanced LogReg log-odds with the minority class on the positive tail."),
    ("03_signature_match.png",          "03_w1_signature_sweep.png",
     "480-cell W1 encoder signature sweep across head families x D in {4..30} x class balance x class separation. Best fit at (LogReg, D=16, [0.8, 0.2], sep=0.5), W1 = 0.0589, KS p = 0.43."),
    ("05_theoretical_floor.png",        "04_theoretical_floor.png",
     "Cramer-Rao and Fano-inequality SRMSE floors as a function of measured H(X|Z) for D=16; both floors sit within 1.7 % of the all-zeros baseline."),
    ("06_ablation_bar.png",             "05_ablation_18_variants.png",
     "18-variant algorithmic ablation x 100 Monte-Carlo seeds. Shipped variant in red, 2024 literature refinements in blue, calibration / single-channel ablations in green. No variant statistically beats the all-zeros baseline."),
    ("20_gmm_fit.png",                  "06_gmm_two_component_fit.png",
     "Two-component GMM EM fit on Z: weights [0.40, 0.60], means {-0.80, +1.47}, variances {2.30, 5.03}, BIC drop 159 from k=1 to k=2 confirming the latent class structure."),
    ("12_qq_normal.png",                "07_qq_normal.png",
     "Q-Q plot of Z against fitted Gaussian: upper-tail divergence confirms the magnitude-leak attack surface documented in paper section 10.2."),
    ("16_welch_psd.png",                "08_welch_psd.png",
     "Welch power spectral density of Z (Hann window, 50 % overlap): flat spectrum confirms white-noise-like row independence underpinning the row-wise reconstruction premise."),
]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    captions = []
    for src_name, dst_name, caption in ORDER:
        src = FIG / src_name
        if not src.exists():
            print(f"  SKIP missing: {src}")
            continue
        dst = DST / dst_name
        shutil.copy2(src, dst)
        size = Image.open(dst).size
        print(f"  copied {src_name:30s} -> {dst_name:35s} ({size[0]}x{size[1]})")
        captions.append((dst_name, caption))

    captions_path = DST / "captions.md"
    captions_path.write_text(
        "# Kaggle writeup media-gallery captions\n\n"
        "Upload these images in order. Paste the caption text into each\n"
        "image's caption field on the writeup form.\n\n"
        + "\n".join(
            f"## `{name}`\n\n{cap}\n" for name, cap in captions
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote captions to {captions_path}")


if __name__ == "__main__":
    main()
