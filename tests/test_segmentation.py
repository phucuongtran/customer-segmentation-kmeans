# -*- coding: utf-8 -*-
"""Tests cho tách B2B, đánh số cụm ổn định, bảng gán cụm và KPI hồ sơ cụm."""

import numpy as np
import pandas as pd
import pytest

from clustering_library.profiling import compute_customer_kpis, summarise_clusters
from clustering_library.segmentation import (
    B2B_CLUSTER_ID,
    CLUSTER_NAMES,
    build_assignments,
    canonical_labels,
    canonical_order,
    cluster_medians,
    split_b2b,
    sweep_k,
)


def test_split_b2b_uses_strict_threshold():
    raw = pd.DataFrame({"SKU_HHI": [0.1, 0.5, 0.9]}, index=pd.Index([1, 2, 3], name="CustomerID"))
    b2b, retail = split_b2b(raw, threshold=0.5)
    assert list(b2b) == [3]
    assert list(retail) == [1, 2]


def _medians():
    # nhãn thô 0..3 với hồ sơ được thiết kế sẵn
    return pd.DataFrame(
        {
            "ReturnRate":       [0.0, 0.0, 2.0, 0.0],  # nhãn thô 2 → C1
            "BasketSizeCV":     [0.5, -1.0, 0.5, 0.4],  # nhãn thô 1 → C2
            "MonthlyOrderRate": [0.1, 0.0, 0.0, 1.5],  # nhãn thô 3 → C0
        },
        index=[0, 1, 2, 3],
    )


def test_canonical_order():
    # [C0, C1, C2, C3] = [3, 2, 1, 0]
    assert canonical_order(_medians()) == [3, 2, 1, 0]


def test_canonical_labels_maps_each_raw_label():
    med = _medians()
    raw_labels = np.repeat([0, 1, 2, 3], 2)
    X = med.loc[raw_labels].reset_index(drop=True)  # mỗi điểm bằng đúng trung vị cụm
    assert list(canonical_labels(X, raw_labels)) == [3, 3, 2, 2, 1, 1, 0, 0]


def test_cluster_medians_accepts_array_or_frame():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    labels = np.array([0, 0, 1])
    from_array = cluster_medians(X, labels, ["a", "b"])
    from_frame = cluster_medians(pd.DataFrame(X, columns=["a", "b"], index=[7, 8, 9]), labels)
    pd.testing.assert_frame_equal(from_array, from_frame)
    assert from_array.loc[0, "a"] == 2.0


def test_build_assignments_marks_b2b():
    out = build_assignments(pd.Index([1, 2, 3]), pd.Index([1, 3]), np.array([0, 2]))
    assert list(out.columns) == ["CustomerID", "cluster", "segment"]
    row = out.set_index("CustomerID")
    assert row.loc[2, "cluster"] == B2B_CLUSTER_ID
    assert row.loc[3, "segment"] == CLUSTER_NAMES[2]


def test_sweep_k_returns_all_metrics():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.1, size=(20, 2)) for loc in (0, 5, 10)])
    metrics = sweep_k(X, [2, 3], verbose=False)
    assert list(metrics.index) == [2, 3]
    assert {"Silhouette", "DaviesBouldin", "CalinskiHarabasz", "Composite"} <= set(metrics.columns)
    assert metrics["Silhouette"].idxmax() == 3


def test_customer_kpis_and_summary():
    tx = pd.DataFrame({
        "CustomerID": [1, 1, 2],
        "InvoiceNo": ["A", "B", "C"],
        "StockCode": ["X", "X", "Y"],
        "Quantity": [2, 4, 1],
        "TotalPrice": [10.0, 30.0, 5.0],
    })
    kpi = compute_customer_kpis(tx)
    assert kpi.loc[1, "TotalRevenue"] == 40.0
    assert kpi.loc[1, "AOV"] == 20.0
    assert kpi.loc[1, "AvgBasketItems"] == 3.0
    assert kpi.loc[2, "NumUniqueSKU"] == 1

    profiles = kpi.assign(cluster=[0, 1])
    summary = summarise_clusters(profiles, CLUSTER_NAMES, metrics=["TotalRevenue", "NumOrders"])
    assert summary["pct_revenue"].sum() == pytest.approx(100.0)
    assert summary.index[0] == f"C0 · {CLUSTER_NAMES[0]}"
