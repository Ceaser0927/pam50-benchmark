"""Generate small fake Xena-format files to smoke-test the pipeline (NOT real data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
sizes = {"LumA": 400, "LumB": 180, "Her2": 70, "Basal": 160, "Normal": 35}
genes = [g if g not in ("NUF2", "NDC80") else C.GENE_ALIASES[g][0] for g in C.PAM50_GENES]
genes += [f"FAKE{i}" for i in range(30)]
centroids = {c: rng.normal(8, 1.5, len(genes)) for c in sizes}
centroids["LumB"] = centroids["LumA"] + rng.normal(0, 0.6, len(genes))  # LumA/B overlap

cols, labels, X = [], [], []
for c, n in sizes.items():
    for _ in range(n):
        sid = f"TCGA-{rng.integers(10, 99)}-{rng.integers(1000, 9999)}-01"
        cols.append(sid); labels.append(c)
        X.append(centroids[c] + rng.normal(0, 1.2, len(genes)))
cols += ["TCGA-AA-0001-11", "TCGA-AA-0002-01", cols[0][:12] + "-01"]  # normal, unlabeled, dup
labels += ["LumA", "", labels[0]]
X += [centroids["Normal"]] * 3
expr = pd.DataFrame(np.array(X).T, index=genes, columns=cols)
expr = expr.loc[:, ~expr.columns.duplicated()]
expr.iloc[0, 5] = np.nan
expr.index.name = "sample"
expr.to_csv(out / C.EXPR_FILE, sep="\t", compression="gzip")
pd.DataFrame({C.LABEL_COL: labels}, index=pd.Index(cols, name="sampleID")) \
    .loc[lambda d: ~d.index.duplicated()].to_csv(out / C.PHENO_FILE, sep="\t")
print("wrote", out)
