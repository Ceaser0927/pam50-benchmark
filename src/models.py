"""Model definitions. Every model is a Pipeline (impute -> scale -> classifier) so that
preprocessing is re-fitted inside each CV fold and never sees validation/test data."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config as C


def _pipe(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", clf)])


def get_models(n_classes):
    return {
        "Logistic Regression": _pipe(LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=C.SEED)),
        "Random Forest": _pipe(RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=C.SEED, n_jobs=1)),
        "SVM (RBF)": _pipe(SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=C.SEED)),
        # XGBoost has no class_weight; balanced sample weights are passed at fit time
        "XGBoost": _pipe(XGBClassifier(
            objective="multi:softprob", num_class=n_classes, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
            random_state=C.SEED, n_jobs=1)),
    }


NEEDS_SAMPLE_WEIGHT = {"XGBoost"}
