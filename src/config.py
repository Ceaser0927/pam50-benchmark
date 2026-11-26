"""Central configuration: every seed, split and hyper-parameter grid lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

# Input files (UCSC Xena, TCGA Hub, cohort "TCGA Breast Cancer (BRCA)")
EXPR_FILE = "HiSeqV2.gz"            # TCGA.BRCA.sampleMap/HiSeqV2  (log2(RSEM norm_count + 1))
PHENO_FILE = "BRCA_clinicalMatrix"  # TCGA.BRCA.sampleMap/BRCA_clinicalMatrix
LABEL_COL = "PAM50Call_RNAseq"

SEED = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
CV_REPEATS = 5
N_BOOTSTRAP = 2000
N_PERM_REPEATS = 30
N_JOBS = -1

CLASS_ORDER = ["LumA", "LumB", "Her2", "Basal", "Normal"]
CLASS_NAMES = {"LumA": "Luminal A", "LumB": "Luminal B", "Her2": "HER2-enriched",
               "Basal": "Basal-like", "Normal": "Normal-like"}

# PAM50 genes (Parker et al., 2009) with older HGNC symbols used in some TCGA annotations
PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6",
    "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1",
    "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK",
    "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6",
    "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T",
]
GENE_ALIASES = {"NUF2": ["CDCA1"], "NDC80": ["KNTC2"], "ORC6": ["ORC6L"],
                "CXXC5": ["RINF"], "ACTR3B": ["ARP11"], "BIRC5": ["API4"]}

PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 1.0, 10.0]},
    "Random Forest": {"clf__max_features": ["sqrt", 0.3],
                      "clf__min_samples_leaf": [1, 3, 5, 10, 20]},
    "SVM (RBF)": {"clf__C": [0.1, 1.0, 10.0, 100.0],
                  "clf__gamma": ["scale", 0.01, 0.001, 1e-4]},
    "XGBoost": {"clf__max_depth": [1, 2, 4],
                "clf__learning_rate": [0.03, 0.1, 0.3],
                "clf__n_estimators": [100, 200, 400]},
}
