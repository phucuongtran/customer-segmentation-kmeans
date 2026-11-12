# -*- coding: utf-8 -*-
"""Kiểm tra giá trị từng công thức đặc trưng trên dữ liệu nhỏ tính tay được."""

import numpy as np
import pandas as pd
import pytest

from clustering_library import feature_catalog
from clustering_library.customer_features import (
    basket_behaviour,
    compute_all_candidates,
    gini,
    product_concentration,
    purchase_rhythm,
    quarter_concentration,
    return_behaviour,
    spend_acceleration,
    volume_bulk,
)


def _tx(rows):
    """rows: (CustomerID, InvoiceNo, date, StockCode, Quantity, UnitPrice)."""
    df = pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate",
                                     "StockCode", "Quantity", "UnitPrice"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


@pytest.fixture
def tx():
    # Khách 1: 3 hoá đơn cách nhau 10 và 30 ngày; khách 2: 1 hoá đơn, 1 SKU.
    return _tx([
        (1, "A1", "2011-01-01", "X", 1, 10.0),
        (1, "A1", "2011-01-01", "Y", 2, 5.0),
        (1, "A2", "2011-01-11", "X", 12, 1.0),
        (1, "A3", "2011-02-10", "Z", 3, 2.0),
        (2, "B1", "2011-02-05", "X", 4, 2.5),
    ])


def test_gini_edge_cases():
    assert gini(np.array([5.0])) == 0.0
    assert gini(np.array([0.0, 0.0])) == 0.0
    assert gini(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert gini(np.array([0.0, 0.0, 1.0])) == pytest.approx(2 / 3)


def test_purchase_rhythm(tx):
    f = purchase_rhythm(tx)
    # snapshot = 2011-02-10 + 1 ngày
    assert f.loc[1, "Recency"] == 1
    assert f.loc[2, "Recency"] == 6
    assert f.loc[1, "ActiveSpanDays"] == 40
    # 3 hoá đơn / (40/30 tháng)
    assert f.loc[1, "MonthlyOrderRate"] == pytest.approx(3 / (40 / 30))
    # khách 1 hoá đơn: span 0 → mẫu số được chặn dưới bằng 1 tháng
    assert f.loc[2, "MonthlyOrderRate"] == 1.0
    gaps = np.array([10.0, 30.0])
    assert f.loc[1, "InterPurchaseCV"] == pytest.approx(gaps.std(ddof=1) / gaps.mean())
    assert f.loc[2, "InterPurchaseCV"] == 0.0


def test_basket_behaviour(tx):
    f = basket_behaviour(tx)
    # A1 {X,Y}: 2/2 mới; A2 {X}: 0/1 mới; A3 {Z}: 1/1 mới → TB = 2/3
    assert f.loc[1, "NewSKURate"] == pytest.approx(2 / 3)
    # X xuất hiện ở 2 hoá đơn, Y và Z ở 1 → 1/3
    assert f.loc[1, "RepeatSKUFraction"] == pytest.approx(1 / 3)
    assert f.loc[2, "BasketSizeCV"] == 0.0


def test_volume_and_concentration(tx):
    vol = volume_bulk(tx)
    # khách 1 có 4 dòng, 1 dòng Quantity >= 12
    assert vol.loc[1, "BulkLineRate"] == pytest.approx(0.25)
    # giá trị hoá đơn: A1=20, A2=12, A3=6 → max/mean = 20/(38/3)
    assert vol.loc[1, "BurstIndex"] == pytest.approx(20 / (38 / 3))
    assert vol.loc[1, "MedianBasketQty"] == 3  # tổng qty mỗi hoá đơn: 3, 12, 3

    conc = product_concentration(tx)
    # chi tiêu theo SKU khách 1: X=22, Y=10, Z=6 trên tổng 38
    expected_hhi = (22 / 38) ** 2 + (10 / 38) ** 2 + (6 / 38) ** 2
    assert conc.loc[1, "SKU_HHI"] == pytest.approx(expected_hhi)
    assert conc.loc[2, "SKU_HHI"] == pytest.approx(1.0)
    assert conc.loc[1, "MaxSpendShare"] == pytest.approx(20 / 38)


def test_return_behaviour_counts_cancelled_invoices():
    raw = pd.DataFrame({
        "InvoiceNo": ["100", "101", "C102", "200"],
        "CustomerID": [1.0, 1.0, 1.0, 2.0],
        "Country": ["United Kingdom"] * 4,
        "Quantity": [2, 2, -1, 5],
        "UnitPrice": [10.0, 10.0, 10.0, 1.0],
    })
    f = return_behaviour(raw, pd.Index([1, 2, 3]))
    assert f.loc[1, "ReturnRate"] == pytest.approx(1 / 3)
    assert f.loc[1, "ReturnValueRate"] == pytest.approx(10 / 50)
    assert f.loc[2, "ReturnRate"] == 0.0
    assert f.loc[3, "ReturnRate"] == 0.0  # không có trong dữ liệu thô


def test_return_behaviour_without_raw_is_zero():
    f = return_behaviour(None, pd.Index([1, 2]))
    assert (f.to_numpy() == 0).all()


def test_spend_acceleration_and_quarter_concentration():
    tx = _tx([
        (1, "A1", "2011-01-15", "X", 1, 10.0),
        (1, "A2", "2011-02-15", "X", 1, 10.0),
        (1, "A3", "2011-03-15", "X", 1, 10.0),
        (1, "A4", "2011-04-15", "X", 1, 40.0),
        (1, "A5", "2011-05-15", "X", 1, 40.0),
        (1, "A6", "2011-06-15", "X", 1, 40.0),
        (2, "B1", "2011-06-15", "X", 1, 5.0),
    ])
    ids = pd.Index([1, 2])
    acc = spend_acceleration(tx, ids)
    assert acc.loc[1] == pytest.approx(3.0)   # (40-10)/10 = 3, nằm đúng biên cắt
    assert acc.loc[2] == 0.0                  # < 3 tháng có mua

    qc = quarter_concentration(tx, ids)
    assert qc.loc[1] == pytest.approx(120 / 150)  # Q2 = 120 trên tổng 150
    assert qc.loc[2] == 1.0


def test_compute_all_candidates_columns(tx):
    candidates = compute_all_candidates(tx)
    assert list(candidates.columns) == feature_catalog.CANDIDATES
    assert not candidates.isnull().values.any()
