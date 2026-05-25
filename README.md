# ML Classifiers & Recommender Systems Demo

End-to-end Python pipeline for building ML classifiers and collaborative-filtering recommenders.

## Architecture

```
raw_data.csv
    └── preprocess.py   (label encoding, train/test split)
    └── classifier.py   (LogisticRegression, RandomForest, XGBoost ensemble)
    └── recommender.py  (user-item matrix, SVD-based collaborative filtering)
    └── main.py         (CLI entry point — runs full pipeline)
```

## Setup

```bash
pip install -r requirements.txt
python main.py --mode classifier --data data/sample.csv
python main.py --mode recommender --data data/ratings.csv
```

## Outputs

- `outputs/classifier_report.json` — precision, recall, F1 per class
- `outputs/confusion_matrix.png` — heatmap
- `outputs/top10_recommendations.csv` — per-user top-10 item list
