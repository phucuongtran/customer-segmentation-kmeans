# -*- coding: utf-8 -*-
"""Tests cho bộ lọc tương quan và trọng số nhóm."""

import numpy as np
import pandas as pd
import pytest

from clustering_library.preprocessing import family_weights, filter_correlated, to_model_space


def _frame():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    return pd.DataFrame({
        "Recency": a,
        "ActiveSpanDays": a + rng.normal(scale=0.01, size=200),  # gần trùng Recency
        "AOV": rng.normal(size=200),
        "PriceCV": np.ones(200),                                  # hằng số
    })


def test_filter_drops_duplicate_and_constant():
    kept = filter_correlated(_frame(), threshold=0.85)
    assert "PriceCV" not in kept
    assert "AOV" in kept
    assert len({"Recency", "ActiveSpanDays"} & set(kept)) == 1


def test_filter_never_drops_protected():
    kept = filter_correlated(_frame(), threshold=0.85, protected=["ActiveSpanDays"])
    assert "ActiveSpanDays" in kept
    assert "Recency" not in kept


def test_family_weights_equalise_families():
    weights = family_weights(["Recency", "MonthlyOrderRate", "AOV", "ReturnRate"])
    assert weights["Recency"] == pytest.approx(1 / np.sqrt(2))   # 2 đặc trưng Purchase Rhythm
    assert weights["AOV"] == pytest.approx(1.0)
    # tổng bình phương trọng số mỗi nhóm = 1
    assert weights["Recency"] ** 2 + weights["MonthlyOrderRate"] ** 2 == pytest.approx(1.0)


def test_to_model_space_is_clipped_and_shaped():
    frame = _frame().drop(columns="PriceCV")
    out = to_model_space(frame)
    assert out.shape == frame.shape
    assert out.abs().max().max() <= 3.0
