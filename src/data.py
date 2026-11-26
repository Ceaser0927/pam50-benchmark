"""Load TCGA-BRCA expression + PAM50 labels, apply QC filters, keep PAM50 genes."""
import json
import pandas as pd

from . import config as C


def load_expression(path):
    """Xena matrix: rows = genes, columns = samples. Returns samples x genes."""
    expr = pd.read_csv(path, sep="\t", index_col=0, compression="infer")
    expr = expr[~expr.index.duplicated(keep="first")]
    return expr.T


def load_labels(path):
    ph = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    return ph[C.LABEL_COL]


def select_pam50(expr):
    """Map the 50 PAM50 genes onto the matrix, resolving known aliases."""
    cols, missing, renamed = {}, [], {}
    for g in C.PAM50_GENES:
        hit = g if g in expr.columns else next(
            (a for a in C.GENE_ALIASES.get(g, []) if a in expr.columns), None)
        if hit is None:
            missing.append(g)
        else:
            cols[g] = hit
            if hit != g:
                renamed[g] = hit
    X = expr[list(cols.values())].copy()
    X.columns = list(cols.keys())
    return X, missing, renamed


def build_cohort(data_dir=C.DATA_DIR, out_dir=C.RESULTS_DIR):
    expr = load_expression(data_dir / C.EXPR_FILE)
    labels = load_labels(data_dir / C.PHENO_FILE)
    log = {"expression_samples": int(expr.shape[0]), "expression_genes": int(expr.shape[1]),
           "phenotype_samples": int(labels.shape[0])}

    # 1. samples present in both files
    common = expr.index.intersection(labels.index)
    expr, labels = expr.loc[common], labels.loc[common]
    log["in_both_files"] = len(common)

    # 2. primary tumours only (TCGA sample-type code 01; drops normals 11 and metastases 06)
    is_primary = expr.index.str[13:15] == "01"
    expr, labels = expr[is_primary], labels[is_primary]
    log["primary_tumour"] = int(is_primary.sum())

    # 3. valid PAM50 label
    labels = labels.where(labels.isin(C.CLASS_ORDER))
    ok = labels.notna()
    expr, labels = expr[ok], labels[ok]
    log["valid_pam50_label"] = int(ok.sum())

    # 4. one sample per patient (first 12 characters of the barcode = patient)
    patient = expr.index.str[:12]
    keep = ~pd.Index(patient).duplicated(keep="first")
    log["duplicate_patient_samples_removed"] = int((~keep).sum())
    expr, labels = expr[keep], labels[keep]

    # 5. fixed PAM50 feature set
    X, missing, renamed = select_pam50(expr)
    log.update(final_samples=int(X.shape[0]), pam50_genes_found=int(X.shape[1]),
               pam50_genes_missing=missing, aliases_used=renamed,
               missing_values=int(X.isna().sum().sum()),
               class_counts={k: int(v) for k, v in labels.value_counts()
                             .reindex(C.CLASS_ORDER).fillna(0).items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cohort_log.json").write_text(json.dumps(log, indent=2))
    return X, labels, log
