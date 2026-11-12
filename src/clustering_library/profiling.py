# -*- coding: utf-8 -*-
"""Hồ sơ cụm ở đơn vị kinh doanh (bảng Anh, số đơn, số ngày) thay vì model space."""

from typing import Dict, List

import pandas as pd

# Đặc trưng hành vi thô dùng để mô tả cụm (lấy từ FeatureEngineer.compute_candidates()).
PROFILE_BEHAVIOUR_FEATURES: List[str] = [
    "Recency", "MonthlyOrderRate", "ReturnRate", "QuarterConcentration", "RepeatSKUFraction",
]

# Các chỉ số đưa vào bảng tóm tắt theo cụm (theo thứ tự hiển thị).
PROFILE_METRICS: List[str] = [
    "TotalRevenue", "NumOrders", "AOV", "AvgBasketItems", "NumUniqueSKU",
    "Recency", "MonthlyOrderRate", "ReturnRate", "RepeatSKUFraction", "QuarterConcentration",
]


def compute_customer_kpis(transactions: pd.DataFrame) -> pd.DataFrame:
    """KPI kinh doanh dễ đọc cho từng khách (index = CustomerID).

    - ``TotalRevenue``  : tổng TotalPrice (LTV / Monetary, doanh thu gộp)
    - ``NumOrders``     : số hoá đơn phân biệt (Frequency)
    - ``NumUniqueSKU``  : số sản phẩm khác nhau đã mua
    - ``TotalItems``    : tổng Quantity
    - ``AOV``           : TotalRevenue / NumOrders
    - ``AvgBasketItems``: TotalItems / NumOrders
    """
    by_customer = transactions.groupby("CustomerID")
    kpis = pd.DataFrame({
        "TotalRevenue": by_customer["TotalPrice"].sum(),
        "NumOrders": by_customer["InvoiceNo"].nunique(),
        "NumUniqueSKU": by_customer["StockCode"].nunique(),
        "TotalItems": by_customer["Quantity"].sum(),
    })
    kpis["AOV"] = kpis["TotalRevenue"] / kpis["NumOrders"]
    kpis["AvgBasketItems"] = kpis["TotalItems"] / kpis["NumOrders"]
    return kpis


def summarise_clusters(
    profiles: pd.DataFrame,
    segment_names: Dict[int, str],
    metrics: List[str] = PROFILE_METRICS,
) -> pd.DataFrame:
    """Bảng tóm tắt theo cụm: quy mô, đóng góp doanh thu và trung vị từng chỉ số.

    ``profiles`` cần cột ``cluster``, ``TotalRevenue`` và các cột trong ``metrics``.
    Trung vị bền với outlier; riêng doanh thu dùng TỔNG để đo đóng góp.
    """
    by_cluster = profiles.groupby("cluster")
    summary = by_cluster[metrics].median()

    n = by_cluster.size()
    revenue = by_cluster["TotalRevenue"].sum()
    summary.insert(0, "n", n)
    summary.insert(1, "pct_customers", (n / n.sum() * 100).round(1))
    summary.insert(2, "revenue_total", revenue.round(0))
    summary.insert(3, "pct_revenue", (revenue / revenue.sum() * 100).round(1))

    summary.index = [f"C{c} · {segment_names[c]}" for c in summary.index]
    return summary
