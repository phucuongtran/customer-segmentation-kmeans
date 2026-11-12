# -*- coding: utf-8 -*-
"""Biểu đồ EDA cho notebook 01."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visual_style import BLUE, CLUSTER_COLORS, SEQUENTIAL_CMAP


def _finish(title: str, xlabel: str = None, ylabel: str = None) -> None:
    """Đặt tiêu đề/nhãn trục cho figure hiện tại rồi hiển thị."""
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def _hist(series: pd.Series, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(series, bins=30, kde=True, color=BLUE)
    _finish(title, xlabel, ylabel)


def _below_p99(series: pd.Series) -> pd.Series:
    """Bỏ 1% giá trị lớn nhất để biểu đồ phân phối lệch phải dễ đọc hơn."""
    return series[series < series.quantile(0.99)]


class DataVisualizer:
    """Các biểu đồ khám phá dữ liệu giao dịch và RFM."""

    def __init__(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette(CLUSTER_COLORS)

    def plot_revenue_over_time(self, df: pd.DataFrame) -> None:
        """Doanh thu theo ngày và theo tháng (cần cột InvoiceDate, TotalPrice)."""
        plt.figure(figsize=(12, 5))
        df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().plot(color=BLUE)
        _finish("Daily Revenue", "Date", "Revenue (GBP)")

        plt.figure(figsize=(12, 5))
        monthly = df.groupby(pd.Grouper(key="InvoiceDate", freq="ME"))["TotalPrice"].sum()
        monthly.plot(kind="bar", color=BLUE)
        plt.xticks(rotation=45)
        _finish("Monthly Revenue", "Month", "Revenue (GBP)")

    def plot_time_patterns(self, df: pd.DataFrame) -> None:
        """Heatmap số dòng giao dịch theo ngày trong tuần × giờ (cần DayOfWeek, HourOfDay)."""
        plt.figure(figsize=(12, 5))
        counts = df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        sns.heatmap(counts, cmap=SEQUENTIAL_CMAP)
        _finish("Purchase Activity by Day and Hour", "Hour of Day", "Day of Week (0=Monday, 6=Sunday)")

    def plot_product_analysis(self, df: pd.DataFrame, top_n: int = 10) -> None:
        """Top sản phẩm theo số lượng bán và theo doanh thu."""
        for value_col, title, xlabel in [
            ("Quantity", f"Top {top_n} Products by Units Sold", "Units Sold"),
            ("TotalPrice", f"Top {top_n} Products by Revenue", "Revenue (GBP)"),
        ]:
            top = df.groupby("Description")[value_col].sum().nlargest(top_n)
            plt.figure(figsize=(12, 5))
            sns.barplot(x=top.values, y=top.index, color=BLUE)
            _finish(title, xlabel)

    def plot_customer_distribution(self, df: pd.DataFrame) -> None:
        """Phân phối số hoá đơn và tổng chi tiêu mỗi khách."""
        by_customer = df.groupby("CustomerID")
        _hist(by_customer["InvoiceNo"].nunique(), "Distribution of Transactions per Customer",
              "Number of Transactions", "Number of Customers")
        _hist(_below_p99(by_customer["TotalPrice"].sum()), "Distribution of Total Spend per Customer",
              "Total Spend (GBP)", "Number of Customers")

    def plot_rfm_analysis(self, rfm_data: pd.DataFrame) -> None:
        """Phân phối Recency, Frequency, Monetary (Monetary bỏ 1% lớn nhất)."""
        panels = [
            (rfm_data["Recency"], "Recency Distribution (Days Since Last Purchase)", "Days"),
            (rfm_data["Frequency"], "Frequency Distribution (Number of Transactions)", "Number of Transactions"),
            (_below_p99(rfm_data["Monetary"]), "Monetary Distribution (Total Spend)", "Total Spend (GBP)"),
        ]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        for ax, (series, title, xlabel) in zip(axes, panels):
            sns.histplot(series, bins=30, kde=True, ax=ax, color=BLUE)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
        plt.tight_layout()
        plt.show()
