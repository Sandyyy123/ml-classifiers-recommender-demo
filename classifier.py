"""
Multi-model classifier pipeline with ensemble voting.
Supports: LogisticRegression, RandomForest, XGBoost (if installed).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class ClassifierPipeline:
    def __init__(self, target_col: str = "label", test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None

    def _build_ensemble(self):
        estimators = [
            ("lr", LogisticRegression(max_iter=500, random_state=self.random_state)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
        ]
        if HAS_XGB:
            estimators.append(("xgb", XGBClassifier(
                n_estimators=100, random_state=self.random_state,
                use_label_encoder=False, eval_metric="mlogloss", verbosity=0
            )))
        return VotingClassifier(estimators=estimators, voting="soft")

    def fit_evaluate(self, df: pd.DataFrame) -> dict:
        feature_cols = [c for c in df.columns if c != self.target_col]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0).values
        y = self.le.fit_transform(df[self.target_col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = self._build_ensemble()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="f1_macro")

        return {
            "summary": {
                "accuracy": round(report["accuracy"], 4),
                "macro_f1": round(report["macro avg"]["f1-score"], 4),
                "cv_f1_mean": round(float(cv_scores.mean()), 4),
                "cv_f1_std": round(float(cv_scores.std()), 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "models": ["LogisticRegression", "RandomForest"] + (["XGBoost"] if HAS_XGB else []),
            },
            "per_class": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")},
            "confusion_matrix": cm,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        encoded = self.model.predict(X_scaled)
        return self.le.inverse_transform(encoded)
