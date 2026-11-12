# -*- coding: utf-8 -*-
"""Tính 20 đặc trưng hành vi ở cấp khách hàng từ dữ liệu giao dịch.

Mỗi hàm nhận DataFrame giao dịch đã làm sạch (một dòng = một dòng hàng trên hoá đơn,
có các cột CustomerID, InvoiceNo, InvoiceDate, StockCode, Quantity, UnitPrice,
TotalPrice) và trả về các cột đặc trưng, index là CustomerID.

Quy ước chung:
- "CV" (coefficient of variation) = độ lệch chuẩn / trung bình. Khi chỉ có một quan
  sát (std = NaN) hoặc trung bình bằng 0, CV được gán 0.
- Hàm thuần (không đọc file, không giữ trạng thái) để dễ test và tái sử dụng.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from clustering_library.feature_catalog import BULK_QTY_THRESHOLD, CANDIDATES

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def gini(values: np.ndarray) -> float:
    """Hệ số Gini của mảng không âm; trả về 0 nếu chỉ có 1 giá trị hoặc tổng bằng 0."""
    vals = np.sort(values)
    n = len(vals)
    if n <= 1 or vals.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * vals).sum() - (n + 1) * vals.sum()) / (n * vals.sum()))


def _cv(grouped: pd.core.groupby.SeriesGroupBy, name: str) -> pd.Series:
    """std / mean theo nhóm; NaN (một quan sát, hoặc 0/0) được gán 0."""
    agg = grouped.agg(["mean", "std"])
    return (agg["std"] / agg["mean"]).fillna(0.0).rename(name)


def _invoice_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng TotalPrice của từng hoá đơn: cột CustomerID, InvoiceNo, InvoiceTotal."""
    return (
        df.groupby(["CustomerID", "InvoiceNo"])["TotalPrice"]
        .sum()
        .reset_index(name="InvoiceTotal")
    )


# --------------------------------------------------------------------------- #
# Feature families                                                             #
# --------------------------------------------------------------------------- #

def purchase_rhythm(df: pd.DataFrame) -> pd.DataFrame:
    """Recency, InterPurchaseCV, ActiveSpanDays, MonthlyOrderRate."""
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    first_last = df.groupby("CustomerID")["InvoiceDate"].agg(["min", "max"])
    recency = (snapshot - first_last["max"]).dt.days.rename("Recency")
    active_span = (first_last["max"] - first_last["min"]).dt.days.rename("ActiveSpanDays")

    invoice_dates = (
        df.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"]
        .min()
        .reset_index()
        .sort_values(["CustomerID", "InvoiceDate"])
    )
    invoice_dates["gap_days"] = invoice_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    inter_purchase_cv = _cv(invoice_dates.groupby("CustomerID")["gap_days"], "InterPurchaseCV")

    n_invoices = df.groupby("CustomerID")["InvoiceNo"].nunique()
    active_months = (active_span / 30).clip(lower=1)
    monthly_order_rate = (n_invoices / active_months).rename("MonthlyOrderRate")

    return pd.concat([recency, inter_purchase_cv, active_span, monthly_order_rate], axis=1)


def spending_shape(df: pd.DataFrame) -> pd.DataFrame:
    """AOV, SpendGini, PriceCV."""
    per_customer = df.groupby("CustomerID").agg(
        total_spend=("TotalPrice", "sum"),
        n_invoices=("InvoiceNo", "nunique"),
    )
    aov = (per_customer["total_spend"] / per_customer["n_invoices"]).rename("AOV")

    spend_gini = (
        _invoice_totals(df)
        .groupby("CustomerID")["InvoiceTotal"]
        .apply(lambda x: gini(x.values))
        .rename("SpendGini")
    )
    price_cv = _cv(df.groupby("CustomerID")["UnitPrice"], "PriceCV")

    return pd.concat([aov, spend_gini, price_cv], axis=1)


def _new_sku_rate(invoice_skus: pd.Series) -> float:
    """Trung bình tỉ lệ SKU 'mới' (chưa từng mua trước đó) trên từng hoá đơn, theo thời gian."""
    seen: set = set()
    rates = []
    for skus in invoice_skus:
        rates.append(len(skus - seen) / len(skus))
        seen.update(skus)
    return float(np.mean(rates)) if rates else 1.0


def basket_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """NewSKURate, RepeatSKUFraction, BasketSizeCV."""
    invoice_skus = (
        df.groupby(["CustomerID", "InvoiceNo"])
        .agg(date=("InvoiceDate", "min"), skus=("StockCode", frozenset))
        .reset_index()
        .sort_values(["CustomerID", "date"])
    )
    new_sku_rate = (
        invoice_skus.groupby("CustomerID")["skus"].apply(_new_sku_rate).rename("NewSKURate")
    )

    invoices_per_sku = df.groupby(["CustomerID", "StockCode"])["InvoiceNo"].nunique()
    repeat_sku_fraction = (
        (invoices_per_sku > 1).groupby(level="CustomerID").mean().rename("RepeatSKUFraction")
    )

    skus_per_invoice = (
        df.groupby(["CustomerID", "InvoiceNo"])["StockCode"].nunique().reset_index(name="n_skus")
    )
    basket_size_cv = _cv(skus_per_invoice.groupby("CustomerID")["n_skus"], "BasketSizeCV")

    return pd.concat([new_sku_rate, repeat_sku_fraction, basket_size_cv], axis=1)


def volume_bulk(df: pd.DataFrame, bulk_qty_threshold: int = BULK_QTY_THRESHOLD) -> pd.DataFrame:
    """BurstIndex, BulkLineRate, QuantityCV, MedianBasketQty."""
    invoice_stats = _invoice_totals(df).groupby("CustomerID")["InvoiceTotal"].agg(["max", "mean"])
    burst_index = (invoice_stats["max"] / invoice_stats["mean"]).rename("BurstIndex")

    is_bulk = (df["Quantity"] >= bulk_qty_threshold).astype(float)
    bulk_line_rate = is_bulk.groupby(df["CustomerID"]).mean().rename("BulkLineRate")

    quantity_cv = _cv(df.groupby("CustomerID")["Quantity"], "QuantityCV")

    median_basket_qty = (
        df.groupby(["CustomerID", "InvoiceNo"])["Quantity"]
        .sum()
        .groupby("CustomerID")
        .median()
        .rename("MedianBasketQty")
    )

    return pd.concat([burst_index, bulk_line_rate, quantity_cv, median_basket_qty], axis=1)


def product_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """SKU_HHI, MaxSpendShare."""
    spend_per_sku = df.groupby(["CustomerID", "StockCode"])["TotalPrice"].sum()
    sku_share = spend_per_sku / spend_per_sku.groupby(level="CustomerID").transform("sum")
    sku_hhi = (sku_share ** 2).groupby(level="CustomerID").sum().rename("SKU_HHI")

    invoices = _invoice_totals(df)
    customer_total = invoices.groupby("CustomerID")["InvoiceTotal"].transform("sum")
    max_spend_share = (
        (invoices["InvoiceTotal"] / customer_total)
        .groupby(invoices["CustomerID"])
        .max()
        .rename("MaxSpendShare")
    )

    return pd.concat([sku_hhi, max_spend_share], axis=1)


def return_behaviour(raw_df: Optional[pd.DataFrame], customer_ids: pd.Index) -> pd.DataFrame:
    """ReturnRate (theo số hoá đơn) và ReturnValueRate (theo giá trị) từ dữ liệu THÔ.

    Dữ liệu đã làm sạch không còn hoá đơn huỷ, nên hàm này cần ``raw_df`` (bản thô,
    còn các hoá đơn 'C...'). Khách không có trong ``raw_df`` hoặc không có hoá đơn huỷ
    nhận 0. Nếu ``raw_df`` là None, cả hai cột bằng 0 (kèm cảnh báo).
    """
    if raw_df is None:
        logger.warning("Không có dữ liệu thô — ReturnRate và ReturnValueRate = 0.")
        zeros = pd.Series(0.0, index=customer_ids)
        return pd.concat([zeros.rename("ReturnRate"), zeros.rename("ReturnValueRate")], axis=1)

    raw_uk = raw_df[
        (raw_df["Country"] == "United Kingdom") & raw_df["CustomerID"].notna()
    ].copy()
    raw_uk["CustomerID"] = raw_uk["CustomerID"].astype(float).astype(int)
    raw_uk["InvoiceNo"] = raw_uk["InvoiceNo"].astype(str)
    raw_uk["is_cancelled"] = raw_uk["InvoiceNo"].str.startswith("C")
    raw_uk["AbsValue"] = (raw_uk["Quantity"] * raw_uk["UnitPrice"]).abs()

    invoice_flags = raw_uk.groupby(["CustomerID", "InvoiceNo"])["is_cancelled"].first()
    counts = invoice_flags.groupby(level="CustomerID").agg(n_cancelled="sum", n_total="count")
    return_rate = (counts["n_cancelled"] / counts["n_total"]).rename("ReturnRate")

    cancelled_value = raw_uk[raw_uk["is_cancelled"]].groupby("CustomerID")["AbsValue"].sum()
    total_value = raw_uk.groupby("CustomerID")["AbsValue"].sum()
    return_value_rate = (cancelled_value / total_value).rename("ReturnValueRate")

    return pd.concat(
        [
            return_rate.reindex(customer_ids).fillna(0.0),
            return_value_rate.reindex(customer_ids).fillna(0.0),
        ],
        axis=1,
    )


def _acceleration(monthly_spend: np.ndarray) -> float:
    """(TB 3 tháng cuối − TB 3 tháng đầu) / |TB 3 tháng đầu|, cắt về [-3, 3]."""
    if len(monthly_spend) < 3:
        return 0.0
    first_avg = monthly_spend[:3].mean()
    last_avg = monthly_spend[-3:].mean()
    raw = (last_avg - first_avg) / (abs(first_avg) + 1e-9)
    return float(np.clip(raw, -3.0, 3.0))


def spend_acceleration(df: pd.DataFrame, customer_ids: pd.Index) -> pd.Series:
    """Xu hướng chi tiêu giữa 3 tháng CÓ MUA đầu tiên và 3 tháng có mua cuối cùng.

    Khách có ít hơn 3 tháng có mua hàng nhận 0. Lưu ý: với 3–5 tháng, hai cửa sổ chồng
    lên nhau (3 tháng thì luôn bằng 0).
    """
    monthly = (
        df.assign(YearMonth=df["InvoiceDate"].dt.to_period("M"))
        .groupby(["CustomerID", "YearMonth"])["TotalPrice"]
        .sum()
    )
    # groupby đã sắp xếp YearMonth tăng dần trong từng khách.
    acceleration = (
        monthly.groupby(level="CustomerID")
        .apply(lambda s: _acceleration(s.values))
        .rename("SpendAcceleration")
    )
    return acceleration.reindex(customer_ids).fillna(0.0)


def quarter_concentration(df: pd.DataFrame, customer_ids: pd.Index) -> pd.Series:
    """Tỉ trọng chi tiêu của quý lớn nhất: 0.25 = rải đều 4 quý, 1.0 = dồn vào một quý.

    Lưu ý: quý được lấy theo ``dt.quarter`` (không tính năm), nên 12/2010 và Q4/2011 cùng
    rơi vào Q4.
    """
    quarter = df["InvoiceDate"].dt.quarter
    customer_total = df.groupby("CustomerID")["TotalPrice"].sum()
    max_quarter_spend = (
        df.groupby(["CustomerID", quarter])["TotalPrice"].sum().groupby(level="CustomerID").max()
    )
    concentration = (max_quarter_spend / customer_total.clip(lower=1e-9)).clip(0.25, 1.0)
    return concentration.rename("QuarterConcentration").reindex(customer_ids).fillna(0.25)


# --------------------------------------------------------------------------- #
# All candidates                                                               #
# --------------------------------------------------------------------------- #

def compute_all_candidates(
    df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
    bulk_qty_threshold: int = BULK_QTY_THRESHOLD,
) -> pd.DataFrame:
    """Tính đủ 20 đặc trưng ứng viên, cột theo thứ tự ``CANDIDATES``.

    NaN còn sót (hiếm) được điền bằng trung vị của cột.
    """
    steps = [
        ("Purchase Rhythm", lambda: purchase_rhythm(df)),
        ("Spending Shape", lambda: spending_shape(df)),
        ("Basket Behaviour (theo dõi SKU tuần tự, có thể mất ~30 giây)", lambda: basket_behaviour(df)),
        ("Volume & Bulk", lambda: volume_bulk(df, bulk_qty_threshold)),
        ("Product Concentration", lambda: product_concentration(df)),
    ]
    blocks = []
    for i, (label, compute) in enumerate(steps, start=1):
        logger.info("[%d/8] %s...", i, label)
        blocks.append(compute())
    base = pd.concat(blocks, axis=1)
    customer_ids = base.index

    logger.info("[6/8] Return behaviour (ReturnRate + ReturnValueRate)...")
    returns = return_behaviour(raw_df, customer_ids)
    logger.info("[7/8] Spend Acceleration...")
    acceleration = spend_acceleration(df, customer_ids)
    logger.info("[8/8] Quarter Concentration...")
    seasonality = quarter_concentration(df, customer_ids)

    candidates = pd.concat([base, returns, acceleration, seasonality], axis=1)[CANDIDATES]

    n_missing = int(candidates.isnull().sum().sum())
    if n_missing:
        logger.warning("%d giá trị NaN — điền bằng trung vị của cột.", n_missing)
        candidates = candidates.fillna(candidates.median())

    logger.info("Đã tính %d ứng viên cho %d khách hàng", len(CANDIDATES), len(candidates))
    return candidates
