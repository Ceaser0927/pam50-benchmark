"""PAM50 subtype benchmark on TCGA-BRCA: end-to-end pipeline.

Usage:
    python run_pipeline.py                     # full analysis, data in data/raw/
    python run_pipeline.py --data-dir X --out-dir Y
"""
import argparse
import json
import platform
import sys
import time
import warnings
from importlib.metadata import version

# SVC(probability=True) is deprecated in scikit-learn 1.9 but still valid (Platt scaling)
warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated.*")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src import config as C
from src.data import build_cohort
from src.evaluate import (METRICS, bootstrap_indices, bootstrap_metrics, ci,
                          compute_metrics, corrected_resampled_ttest)
from src.models import NEEDS_SAMPLE_WEIGHT, get_models

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
MODEL_COLORS = {"Logistic Regression": "#4C72B0", "Random Forest": "#55A868",
                "SVM (RBF)": "#C44E52", "XGBoost": "#8172B2"}


def save_fig(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(data_dir, out):
    t0 = time.time()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    X_df, y_lab, log = build_cohort(data_dir, out)
    classes = [c for c in C.CLASS_ORDER if c in set(y_lab)]
    names = [C.CLASS_NAMES[c] for c in classes]
    y = y_lab.map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = X_df.to_numpy(dtype=float)
    genes = list(X_df.columns)
    K = len(classes)
    print(json.dumps(log, indent=2))

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=C.TEST_SIZE,
                                      stratify=y, random_state=C.SEED)
    Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    pd.DataFrame({"sample": X_df.index[np.r_[idx_tr, idx_te]],
                  "subtype": y_lab.iloc[np.r_[idx_tr, idx_te]].values,
                  "split": ["train"] * len(idx_tr) + ["test"] * len(idx_te)}
                 ).to_csv(out / "tables" / "data_split.csv", index=False)

    # ---------------- Figure 1: cohort ----------------
    counts = pd.DataFrame({"Training": np.bincount(ytr, minlength=K),
                           "Held-out test": np.bincount(yte, minlength=K)}, index=names)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    counts.plot.bar(stacked=True, ax=ax, color=["#4C72B0", "#DD8452"], width=0.7)
    for i, tot in enumerate(counts.sum(axis=1)):
        ax.text(i, tot + 5, f"n={tot}", ha="center", fontsize=11)
    ax.set_ylabel("Number of tumours"); ax.set_xlabel("")
    ax.set_title(f"TCGA-BRCA PAM50 subtypes (N = {len(y)})")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, counts.sum(axis=1).max() * 1.12)
    save_fig(fig, out, "fig1_subtype_distribution")

    # ---------------- tuning with repeated stratified CV ----------------
    cv = RepeatedStratifiedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS,
                                 random_state=C.SEED)
    models = get_models(K)
    best, cv_scores, best_params = {}, {}, {}
    for name, pipe in models.items():
        t = time.time()
        gs = GridSearchCV(pipe, C.PARAM_GRIDS[name], scoring="balanced_accuracy",
                          cv=cv, n_jobs=C.N_JOBS, refit=True)
        fit_kw = {"clf__sample_weight": compute_sample_weight("balanced", ytr)} \
            if name in NEEDS_SAMPLE_WEIGHT else {}
        gs.fit(Xtr, ytr, **fit_kw)
        best[name] = gs.best_estimator_
        best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        n_folds = C.CV_SPLITS * C.CV_REPEATS
        cv_scores[name] = np.array([gs.cv_results_[f"split{i}_test_score"][gs.best_index_]
                                    for i in range(n_folds)])
        print(f"{name:20s} CV bal.acc = {cv_scores[name].mean():.3f} ± "
              f"{cv_scores[name].std():.3f}  {best_params[name]}  ({time.time()-t:.0f}s)")

    # paired CV comparison against the linear baseline (identical folds for every model)
    n_te_fold = len(ytr) // C.CV_SPLITS
    cv_rows = []
    for name, s in cv_scores.items():
        row = {"Model": name, "CV mean": s.mean(), "CV SD": s.std(ddof=1),
               "Best hyper-parameters": json.dumps(best_params[name], default=str)}
        if name != "Logistic Regression":
            d = s - cv_scores["Logistic Regression"]
            tstat, p = corrected_resampled_ttest(d, len(ytr) - n_te_fold, n_te_fold)
            row.update({"Mean diff vs LR": d.mean(), "Corrected t": tstat, "p (corrected)": p})
        cv_rows.append(row)
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(out / "tables" / "cv_summary.csv", index=False)
    pd.DataFrame(cv_scores).to_csv(out / "tables" / "cv_fold_scores.csv", index=False)

    # ---------------- held-out test evaluation ----------------
    preds = {n: m.predict(Xte) for n, m in best.items()}
    probas = {n: m.predict_proba(Xte) for n, m in best.items()}
    boot_idx = bootstrap_indices(yte, C.N_BOOTSTRAP, C.SEED)
    boots, rows = {}, []
    for n in best:
        point = compute_metrics(yte, preds[n], probas[n], K)
        boots[n] = bootstrap_metrics(yte, preds[n], probas[n], K, boot_idx)
        row = {"Model": n}
        for m in METRICS:
            lo, hi = ci(boots[n][m])
            row[m] = point[m]
            row[f"{m} 95% CI"] = f"{lo:.3f}–{hi:.3f}"
        rows.append(row)
    test_table = pd.DataFrame(rows)
    test_table.to_csv(out / "tables" / "test_metrics.csv", index=False)

    # paired bootstrap differences vs logistic regression on the test set
    diff_rows = []
    for n in best:
        if n == "Logistic Regression":
            continue
        for m in ["Balanced accuracy", "Macro F1", "Macro OvR ROC-AUC"]:
            d = boots[n][m] - boots["Logistic Regression"][m]
            lo, hi = ci(d)
            diff_rows.append({"Model": n, "Metric": m, "Diff vs LR (mean)": d.mean(),
                              "95% CI low": lo, "95% CI high": hi,
                              "CI excludes 0": bool(lo > 0 or hi < 0)})
    pd.DataFrame(diff_rows).to_csv(out / "tables" / "test_paired_diff_vs_lr.csv", index=False)

    # markdown table for the report
    md = ["| Model | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for r in rows:
        md.append(f"| {r['Model']} | " + " | ".join(
            f"{r[m]:.3f} ({r[m + ' 95% CI']})" for m in METRICS) + " |")
    (out / "tables" / "test_metrics.md").write_text("\n".join(md) + "\n")

    # ---------------- Figure 2: CV distribution + test CIs ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.2]})
    cv_long = pd.DataFrame(cv_scores).melt(var_name="Model", value_name="Balanced accuracy")
    sns.boxplot(data=cv_long, x="Model", y="Balanced accuracy", hue="Model",
                palette=MODEL_COLORS, ax=axes[0], width=0.55, fliersize=0, legend=False)
    sns.stripplot(data=cv_long, x="Model", y="Balanced accuracy", color="k",
                  size=3, alpha=0.5, ax=axes[0])
    axes[0].set_title(f"A  Repeated CV ({C.CV_REPEATS}×{C.CV_SPLITS}-fold, training set)",
                      loc="left")
    axes[0].set_xlabel(""); axes[0].tick_params(axis="x", rotation=15)
    show = ["Balanced accuracy", "Macro F1", "Macro OvR ROC-AUC"]
    off = np.linspace(-0.27, 0.27, len(best))
    for j, n in enumerate(best):
        pts = [test_table.set_index("Model").loc[n, m] for m in show]
        lohi = np.array([ci(boots[n][m]) for m in show])
        axes[1].errorbar(np.arange(len(show)) + off[j], pts,
                         yerr=[np.array(pts) - lohi[:, 0], lohi[:, 1] - np.array(pts)],
                         fmt="o", capsize=4, color=MODEL_COLORS[n], label=n, ms=7)
    axes[1].set_xticks(range(len(show))); axes[1].set_xticklabels(show)
    axes[1].set_ylabel("Held-out test score")
    axes[1].set_title("B  Held-out test set with bootstrap 95% CI", loc="left")
    axes[1].legend(loc="lower right", fontsize=9, frameon=True)
    save_fig(fig, out, "fig2_model_performance")

    # ---------------- Figure 3: confusion matrices ----------------
    fig, axes = plt.subplots(1, len(best), figsize=(4.2 * len(best), 4.2))
    cm_out = {}
    for ax, n in zip(axes, best):
        cm = confusion_matrix(yte, preds[n], labels=range(K))
        cm_out[n] = cm.tolist()
        norm = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array([[f"{norm[i, j]:.2f}\n({cm[i, j]})" for j in range(K)] for i in range(K)])
        sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax, annot_kws={"size": 9})
        ax.set_title(n); ax.set_xlabel("Predicted")
        ax.set_ylabel("True PAM50 call" if ax is axes[0] else "")
    save_fig(fig, out, "fig3_confusion_matrices")

    # Luminal A / B confusion summary
    iA, iB = classes.index("LumA"), classes.index("LumB")
    lum = {}
    for n, cm in cm_out.items():
        cm = np.array(cm)
        errors = cm.sum() - np.trace(cm)
        lum[n] = {"LumA_predicted_as_LumB": int(cm[iA, iB]),
                  "LumB_predicted_as_LumA": int(cm[iB, iA]),
                  "total_errors": int(errors),
                  "share_of_errors_LumA_LumB": float((cm[iA, iB] + cm[iB, iA]) / max(errors, 1)),
                  "per_class_recall": {classes[k]: float(cm[k, k] / cm[k].sum()) for k in range(K)}}

    # ---------------- Figure 4: ROC curves ----------------
    pal = sns.color_palette("Set1", K)
    fig, axes = plt.subplots(1, len(best), figsize=(4.2 * len(best), 4.2), sharey=True)
    for ax, n in zip(axes, best):
        for k in range(K):
            fpr, tpr, _ = roc_curve(yte == k, probas[n][:, k])
            from sklearn.metrics import auc
            ax.plot(fpr, tpr, color=pal[k], lw=1.8, label=f"{classes[k]} ({auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(n); ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", fontsize=8.5, title="Class (AUC)", title_fontsize=8.5)
    axes[0].set_ylabel("True positive rate")
    save_fig(fig, out, "fig4_roc_curves")

    # ---------------- Figure 5: permutation importance ----------------
    imp = {}
    for n, m in best.items():
        r = permutation_importance(m, Xte, yte, scoring="balanced_accuracy",
                                   n_repeats=C.N_PERM_REPEATS, random_state=C.SEED,
                                   n_jobs=C.N_JOBS)
        imp[n] = pd.Series(r.importances_mean, index=genes)
        pd.DataFrame({"gene": genes, "mean_drop": r.importances_mean,
                      "sd": r.importances_std}).sort_values("mean_drop", ascending=False
                      ).to_csv(out / "tables" / f"perm_importance_{n.split()[0].lower()}.csv",
                               index=False)
    imp_df = pd.DataFrame(imp)
    top = sorted(set().union(*[imp_df[n].nlargest(10).index for n in imp_df]),
                 key=lambda g: -imp_df.loc[g].mean())
    ranks = imp_df.rank(ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 0.33 * len(top) + 1.2))
    sns.heatmap(imp_df.loc[top], annot=ranks.loc[top].astype(int), fmt="d", cmap="rocket_r",
                cbar_kws={"label": "Mean drop in balanced accuracy"}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Permutation importance (cell text = rank within model)")
    ax.set_ylabel("PAM50 gene"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, out, "fig5_permutation_importance")
    imp_df.to_csv(out / "tables" / "perm_importance_all_models.csv")

    # ---------------- provenance ----------------
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor() or platform.machine(),
           **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost",
                                      "scipy", "matplotlib", "seaborn"]}}
    summary = {"cohort": log, "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "classes": classes, "best_params": best_params,
               "cv_mean_balanced_accuracy": {n: float(s.mean()) for n, s in cv_scores.items()},
               "confusion_matrices": cm_out, "luminal_confusion": lum,
               "top10_genes": {n: list(imp_df[n].nlargest(10).index) for n in imp_df},
               "config": {k: getattr(C, k) for k in ["SEED", "TEST_SIZE", "CV_SPLITS",
                          "CV_REPEATS", "N_BOOTSTRAP", "N_PERM_REPEATS"]},
               "environment": env, "runtime_seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(test_table[["Model"] + METRICS].round(3).to_string(index=False))
    print(f"Done in {time.time() - t0:.0f}s. Outputs in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    main(a.data_dir, a.out_dir)
