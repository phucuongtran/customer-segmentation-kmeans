# -*- coding: utf-8 -*-
"""Phân cụm hai giai đoạn: tách nhóm "wholesale-like" bằng quy tắc, rồi K-Means++.

Các hàm ở đây gom lại logic trước đây bị lặp trong notebook 02–05:
tách B2B, quét k, fit K-Means, đánh số cụm ổn định và tính trung vị theo cụm.
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

RANDOM_STATE = 42

# Giai đoạn 1: khách có SKU_HHI (thô) lớn hơn ngưỡng bị tách khỏi bước phân cụm.
B2B_SKU_HHI_THRESHOLD = 0.5
B2B_SEGMENT_NAME = "Wholesale-like Proxy"
B2B_CLUSTER_ID = -1

# Giai đoạn 2: tên 4 cụm retail. Số thứ tự là định danh ổn định — KHÔNG đổi thứ tự,
# màu trong ``visual_style.CLUSTER_COLORS`` và tài liệu bài giảng dựa vào nó.
CLUSTER_NAMES: Dict[int, str] = {
    0: "Seasonal Intensives",
    1: "High-Return Actives",
    2: "One-Time Explorers",
    3: "Occasional Loyalists",
}


# --------------------------------------------------------------------------- #
# Giai đoạn 1                                                                  #
# --------------------------------------------------------------------------- #

def split_b2b(
    raw_features: pd.DataFrame, threshold: float = B2B_SKU_HHI_THRESHOLD
) -> Tuple[pd.Index, pd.Index]:
    """Tách khách theo ``SKU_HHI`` thô. Trả về ``(b2b_ids, retail_ids)``.

    ``raw_features`` phải có index là CustomerID và cột ``SKU_HHI`` ở đơn vị gốc
    (chưa biến đổi).
    """
    is_b2b = raw_features["SKU_HHI"] > threshold
    return raw_features.index[is_b2b], raw_features.index[~is_b2b]


# --------------------------------------------------------------------------- #
# Giai đoạn 2                                                                  #
# --------------------------------------------------------------------------- #

def fit_kmeans(X: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> KMeans:
    """K-Means++ với cấu hình dùng xuyên suốt project (n_init=10, max_iter=300)."""
    return KMeans(
        n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=random_state
    ).fit(X)


def sweep_k(X: np.ndarray, k_values: Iterable[int], verbose: bool = True) -> pd.DataFrame:
    """Fit K-Means cho từng k và tính Silhouette, Davies-Bouldin, Calinski-Harabasz.

    Cột ``Composite`` = 0.5·Silhouette + 0.25·1/(1+DB) + 0.25·CH/CH_max
    (Silhouette có trọng số gấp đôi vì dễ diễn giải nhất).
    """
    rows = []
    for k in k_values:
        labels = fit_kmeans(X, k).labels_
        sil = silhouette_score(X, labels, random_state=RANDOM_STATE)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        rows.append({"k": k, "Silhouette": sil, "DaviesBouldin": db, "CalinskiHarabasz": ch})
        if verbose:
            print(f"k={k}: Sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")

    metrics = pd.DataFrame(rows).set_index("k")
    metrics["Composite"] = (
        0.5 * metrics["Silhouette"]
        + 0.25 / (1 + metrics["DaviesBouldin"])
        + 0.25 * metrics["CalinskiHarabasz"] / metrics["CalinskiHarabasz"].max()
    )
    return metrics


def cluster_medians(X, labels: np.ndarray, columns: List[str] = None) -> pd.DataFrame:
    """Trung vị từng đặc trưng theo cụm (index = nhãn cụm, cột = đặc trưng)."""
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=columns)
    frame = frame.reset_index(drop=True).assign(cluster=np.asarray(labels))
    return frame.groupby("cluster").median()


def canonical_order(medians: pd.DataFrame) -> List[int]:
    """Sắp các nhãn K-Means thô theo định danh nghiệp vụ ổn định.

    K-Means đánh số cụm tuỳ ý. Hàm này trả về ``[nhãn_thô_của_C0, ..., nhãn_thô_của_C3]``
    theo quy tắc trên ``medians`` (trung vị model-space theo cụm):

    - C1 High-Return Actives  : ReturnRate cao nhất
    - C2 One-Time Explorers   : trong phần còn lại, BasketSizeCV thấp nhất (mua 1 lần)
    - C0 Seasonal Intensives  : trong phần còn lại, MonthlyOrderRate cao nhất
    - C3 Occasional Loyalists : cụm còn lại
    """
    remaining = list(medians.index)
    c1 = medians["ReturnRate"].idxmax()
    remaining.remove(c1)
    c2 = medians.loc[remaining, "BasketSizeCV"].idxmin()
    remaining.remove(c2)
    c0 = medians.loc[remaining, "MonthlyOrderRate"].idxmax()
    remaining.remove(c0)
    (c3,) = remaining
    return [int(c0), int(c1), int(c2), int(c3)]


def canonical_labels(X: pd.DataFrame, raw_labels: np.ndarray) -> np.ndarray:
    """Đổi nhãn K-Means thô sang định danh ổn định 0..3 (xem ``canonical_order``)."""
    order = canonical_order(cluster_medians(X, raw_labels))
    mapping = {raw: canonical for canonical, raw in enumerate(order)}
    return np.array([mapping[label] for label in raw_labels])


def build_assignments(
    all_ids: pd.Index, retail_ids: pd.Index, retail_labels: np.ndarray
) -> pd.DataFrame:
    """Bảng gán cụm cho toàn bộ khách: cột CustomerID, cluster, segment.

    Khách không thuộc ``retail_ids`` (nhóm B2B) nhận cluster = -1.
    """
    assignments = pd.DataFrame(
        {"cluster": B2B_CLUSTER_ID, "segment": B2B_SEGMENT_NAME},
        index=pd.Index(all_ids.astype(int), name="CustomerID"),
    )
    assignments.loc[retail_ids.astype(int), "cluster"] = retail_labels
    assignments.loc[retail_ids.astype(int), "segment"] = [CLUSTER_NAMES[c] for c in retail_labels]
    return assignments.reset_index()
