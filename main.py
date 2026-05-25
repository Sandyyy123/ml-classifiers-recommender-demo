#!/usr/bin/env python3
"""
ML Classifiers & Recommender Systems - main entry point.
Usage:
    python main.py --mode classifier --data data/sample.csv
    python main.py --mode recommender --data data/ratings.csv
"""
import argparse, pathlib, json
import pandas as pd
from classifier import ClassifierPipeline
from recommender import RecommenderPipeline

def main():
    parser = argparse.ArgumentParser(description="ML Classifier + Recommender Demo")
    parser.add_argument("--mode", choices=["classifier", "recommender", "both"], default="both")
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--target", default="label", help="Target column for classifier")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    pathlib.Path("outputs").mkdir(exist_ok=True)

    if args.mode in ("classifier", "both"):
        print("\n=== CLASSIFIER PIPELINE ===")
        df = pd.read_csv(args.data) if pathlib.Path(args.data).exists() else _sample_clf_data()
        clf = ClassifierPipeline(target_col=args.target)
        report = clf.fit_evaluate(df)
        out = "outputs/classifier_report.json"
        json.dump(report, open(out, "w"), indent=2)
        print(f"Report saved: {out}")
        print(json.dumps(report["summary"], indent=2))

    if args.mode in ("recommender", "both"):
        print("\n=== RECOMMENDER PIPELINE ===")
        ratings_path = args.data if args.mode == "recommender" else "data/ratings.csv"
        df_r = pd.read_csv(ratings_path) if pathlib.Path(ratings_path).exists() else _sample_ratings()
        rec = RecommenderPipeline(top_n=args.top_n)
        recs = rec.fit_recommend(df_r)
        out = "outputs/top10_recommendations.csv"
        recs.to_csv(out, index=False)
        print(f"Recommendations saved: {out}")
        print(recs.head(20).to_string(index=False))

def _sample_clf_data():
    """Generate demo classification data when no CSV provided."""
    from sklearn.datasets import make_classification
    import numpy as np
    X, y = make_classification(n_samples=500, n_features=10, n_classes=3,
                                n_informative=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    df["label"] = y
    return df

def _sample_ratings():
    """Generate demo ratings matrix."""
    import numpy as np
    np.random.seed(42)
    users = [f"u{i}" for i in range(50)]
    items = [f"item_{j}" for j in range(30)]
    rows = []
    for u in users:
        for item in np.random.choice(items, size=10, replace=False):
            rows.append({"user_id": u, "item_id": item, "rating": np.random.randint(1, 6)})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    main()
