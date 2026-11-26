"""Draw the analysis-pipeline diagram (Figure for report/slides)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path(__file__).resolve().parents[1] / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)

steps = [
    ("Data", "TCGA-BRCA\nRNA-seq\nlog2(RSEM+1)\n+ PAM50 calls", "#DCE6F2"),
    ("QC & filtering", "Primary tumours\nValid PAM50 label\n1 sample / patient", "#DCE6F2"),
    ("Features", "Fixed 50 PAM50\ngenes\n(alias-resolved)", "#DCE6F2"),
    ("Split", "Stratified 80/20\ntrain / held-out\ntest (seed 42)", "#FCE4D6"),
    ("Train + tune", "Impute → scale →\nLR | RF | SVM | XGB\n5×5 repeated CV", "#FCE4D6"),
    ("Evaluate", "Test metrics\nBootstrap 95% CI\nPaired tests vs LR", "#E2EFDA"),
    ("Interpret", "Confusion / ROC\nPermutation\nimportance", "#E2EFDA"),
]
fig, ax = plt.subplots(figsize=(14, 3.0))
w, h, gap = 1.75, 1.75, 0.28
for i, (title, body, col) in enumerate(steps):
    x = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=col, ec="#404040", lw=1.2))
    ax.text(x + w / 2, h - 0.25, title, ha="center", va="center", fontsize=11.5, weight="bold")
    ax.text(x + w / 2, h / 2 - 0.15, body, ha="center", va="center", fontsize=9.5)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.02, h / 2), xytext=(x + w + 0.02, h / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#404040"))
# leakage-safe bracket over train-only steps
x0, x1 = 3 * (w + gap), 4 * (w + gap) + w
ax.plot([x0, x0, x1, x1], [h + 0.15, h + 0.3, h + 0.3, h + 0.15], color="#C00000", lw=1.3)
ax.text((x0 + x1) / 2, h + 0.42, "Preprocessing fitted inside training folds only (no leakage)",
        ha="center", fontsize=9.5, color="#C00000")
ax.set_xlim(-0.1, len(steps) * (w + gap)); ax.set_ylim(-0.1, h + 0.65); ax.axis("off")
for ext in ("png", "pdf"):
    fig.savefig(out / f"fig0_pipeline.{ext}", dpi=300, bbox_inches="tight")
print("saved", out / "fig0_pipeline.png")
