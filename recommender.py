"""
Collaborative filtering recommender using SVD (Truncated SVD).
Input: user-item ratings DataFrame with columns [user_id, item_id, rating].
Output: top-N recommendations per user.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class RecommenderPipeline:
    def __init__(self, n_components: int = 20, top_n: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.top_n = top_n
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_idx = {}
        self.item_idx = {}
        self.idx_to_item = {}
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None

    def _build_matrix(self, df: pd.DataFrame) -> csr_matrix:
        users = df["user_id"].unique().tolist()
        items = df["item_id"].unique().tolist()
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.item_idx = {it: j for j, it in enumerate(items)}
        self.idx_to_item = {j: it for it, j in self.item_idx.items()}

        rows = df["user_id"].map(self.user_idx).values
        cols = df["item_id"].map(self.item_idx).values
        data = df["rating"].values.astype(float)
        return csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    def fit_recommend(self, df: pd.DataFrame) -> pd.DataFrame:
        self.user_item_matrix = self._build_matrix(df)
        n_comp = min(self.n_components, min(self.user_item_matrix.shape) - 1)
        self.svd.n_components = n_comp
        self.user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.item_factors = normalize(self.svd.components_.T)
        user_norms = normalize(self.user_factors)

        records = []
        already_rated = df.groupby("user_id")["item_id"].apply(set).to_dict()

        for user, uid in self.user_idx.items():
            scores = user_norms[uid] @ self.item_factors.T
            rated = already_rated.get(user, set())
            item_scores = [
                (self.idx_to_item[j], float(scores[j]))
                for j in range(len(scores))
                if self.idx_to_item[j] not in rated
            ]
            item_scores.sort(key=lambda x: -x[1])
            for rank, (item, score) in enumerate(item_scores[: self.top_n], 1):
                records.append({"user_id": user, "rank": rank, "item_id": item,
                                 "predicted_score": round(score, 4)})

        return pd.DataFrame(records)

    def similar_items(self, item_id: str, top_n: int = 5) -> list:
        if item_id not in self.item_idx:
            return []
        idx = self.item_idx[item_id]
        scores = self.item_factors @ self.item_factors[idx]
        top = np.argsort(-scores)[1: top_n + 1]
        return [(self.idx_to_item[i], round(float(scores[i]), 4)) for i in top]
