# Benchmarking Machine-Learning Models for PAM50 Subtype Classification (TCGA-BRCA)

Course project, Advanced Bioinformatics I (CSUSB) — Ceaser Kaiwusaier.

**Question:** with the feature set fixed to the 50 PAM50 genes, do nonlinear models
(Random Forest, RBF-SVM, XGBoost) classify breast-cancer intrinsic subtypes
meaningfully better than a linear model (multinomial logistic regression)?

## 1. Requirements

- Python 3.11+ (developed and tested with Python 3.13)
- CPU only; 4 GB RAM is enough. A full run takes about 5–10 min on a 2-core machine.
- Pinned package versions are listed in `requirements.txt`:

| Package | Version |
|---|---|
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |

## 2. Installation

```bash
git clone <this-repo-url> pam50-benchmark
cd pam50-benchmark
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Input data

The data are public and de-identified, so no access application is needed. Download both
files from the UCSC Xena **TCGA Hub**, cohort *TCGA Breast Cancer (BRCA)*
(<https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)>):

| File | Xena dataset ID | Contents |
|---|---|---|
| `HiSeqV2.gz` | `TCGA.BRCA.sampleMap/HiSeqV2` | Gene-level RNA-seq, log2(RSEM normalized count + 1); genes × samples |
| `BRCA_clinicalMatrix` | `TCGA.BRCA.sampleMap/BRCA_clinicalMatrix` | Phenotypes, including the `PAM50Call_RNAseq` labels |

Put both files in `data/raw/` and keep the file names unchanged. `HiSeqV2` may be
saved either gzipped or uncompressed; if it is uncompressed, rename it to `HiSeqV2.gz` or
change `EXPR_FILE` in `src/config.py`.

## 4. Directory structure

```
pam50-benchmark/
├── run_pipeline.py              # entry point: runs the whole analysis
├── requirements.txt
├── src/
│   ├── config.py                # seeds, split sizes, CV design, hyper-parameter grids, gene list
│   ├── data.py                  # loading, QC filters, PAM50 gene selection
│   ├── models.py                # the four leakage-safe sklearn pipelines
│   └── evaluate.py              # metrics, stratified bootstrap, corrected resampled t-test
├── scripts/
│   ├── make_pipeline_diagram.py # draws fig0_pipeline
│   └── make_synthetic_data.py   # fake data for a quick smoke test (not used in the report)
├── data/raw/                    # put the downloaded Xena files here (not committed)
└── results/                     # created by the pipeline
    ├── cohort_log.json          # sample counts after each filtering step
    ├── summary.json             # best hyper-parameters, confusion matrices, top genes, versions
    ├── tables/                  # CSV/Markdown tables
    └── figures/                 # PNG (300 dpi) + PDF
```

## 5. Running the analysis

```bash
python run_pipeline.py                 # full analysis using data/raw -> results/
python scripts/make_pipeline_diagram.py
```

Optional smoke test that needs no download (synthetic data, **not** used for the report):

```bash
python scripts/make_synthetic_data.py data/synthetic
python run_pipeline.py --data-dir data/synthetic --out-dir results_synthetic
```

## 6. What the pipeline does

1. **Cohort construction** (`src/data.py`): match expression and phenotype sample IDs →
   keep primary tumours only (barcode sample-type code `01`) → keep samples with a
   PAM50 call in {LumA, LumB, Her2, Basal, Normal} → keep one sample per patient
   (first 12 characters of the barcode) → keep the 50 PAM50 genes. Outdated gene
   symbols are resolved (for example `CDCA1`→`NUF2`, `KNTC2`→`NDC80`, `ORC6L`→`ORC6`).
   Sample counts after each step are written to `results/cohort_log.json`.
2. **Split:** stratified 80/20 train/test split (`SEED = 42`). The test set is used only
   once, for the final evaluation.
3. **Models** (`src/models.py`): each model is a `Pipeline(median imputation →
   standardisation → classifier)`, so preprocessing is re-fitted inside every CV fold and
   never sees validation or test data. Class imbalance is handled with `class_weight="balanced"`
   (LR, RF, SVM) or balanced sample weights (XGBoost).
4. **Tuning:** grid search (grids in `src/config.py`) with 5-fold × 5-repeat stratified CV
   on the training set, scored by balanced accuracy. All models use identical folds.
5. **Statistics** (`src/evaluate.py`):
   - Per-fold CV differences versus logistic regression, tested with the Nadeau–Bengio
     corrected resampled t-test.
   - Held-out metrics: balanced accuracy, macro precision/recall/F1, and macro one-vs-rest
     ROC-AUC.
   - Stratified bootstrap (2,000 resamples) 95% CIs, plus paired bootstrap differences
     versus logistic regression.
6. **Interpretation:** row-normalised confusion matrices, one-vs-rest ROC curves,
   Luminal A/B confusion counts, and permutation importance on the test set
   (30 repeats, scored by balanced accuracy).

## 7. Expected outputs

| Output | Content |
|---|---|
| `figures/fig0_pipeline` | Workflow diagram |
| `figures/fig1_subtype_distribution` | Subtype counts in the train and test sets |
| `figures/fig2_model_performance` | (A) Repeated-CV distributions; (B) test scores with bootstrap 95% CIs |
| `figures/fig3_confusion_matrices` | Row-normalised confusion matrices (counts shown in brackets) |
| `figures/fig4_roc_curves` | One-vs-rest ROC curves for each model |
| `figures/fig5_permutation_importance` | Top genes across models (cell text = rank within model) |
| `tables/test_metrics.csv` / `.md` | Main model-comparison table with 95% CIs |
| `tables/cv_summary.csv` | CV mean/SD, best hyper-parameters, corrected t-test versus LR |
| `tables/test_paired_diff_vs_lr.csv` | Paired bootstrap differences versus LR |
| `tables/perm_importance_*.csv` | Full gene rankings for each model |
| `tables/data_split.csv` | Exact train/test assignment of every sample |

## 8. Reproducibility notes

- All randomness (split, CV folds, bootstrap, permutation, model seeds) comes from
  `SEED = 42` in `src/config.py`.
- `results/summary.json` records the package versions, platform and runtime of each run.
- Tree-model results can vary by a tiny amount across CPU architectures or thread counts;
  the conclusions do not depend on the third decimal place.
- `SVC(probability=True)` is deprecated in scikit-learn 1.9 but still works. The
  pipeline suppresses that warning.

## 9. Data and software citations

- Parker JS et al. (2009) *J Clin Oncol* 27:1160–1167 — PAM50 gene list.
- The Cancer Genome Atlas Network (2012) *Nature* 490:61–70 — TCGA-BRCA.
- Goldman MJ et al. (2020) *Nat Biotechnol* 38:675–678 — UCSC Xena.
- Pedregosa F et al. (2011) *JMLR* 12:2825–2830 — scikit-learn.
- Chen T, Guestrin C (2016) *KDD* 785–794 — XGBoost.
- Nadeau C, Bengio Y (2003) *Mach Learn* 52:239–281 — corrected resampled t-test.
- Harris CR et al. (2020) NumPy; McKinney W (2010) pandas; Virtanen P et al. (2020) SciPy;
  Hunter JD (2007) Matplotlib; Waskom ML (2021) seaborn.
