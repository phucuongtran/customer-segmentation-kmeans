# -*- coding: utf-8 -*-
"""Đọc và làm sạch dữ liệu giao dịch UCI Online Retail."""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COUNTRY = "United Kingdom"

RAW_DTYPES = {
    "InvoiceNo": "object",
    "StockCode": "object",
    "Description": "object",
    "Quantity": "int64",
    "UnitPrice": "float64",
    "CustomerID": "float64",  # đọc dạng float để giữ NaN, sau đó đổi sang Int64
    "Country": "object",
}


class DataCleaner:
    """Đọc dữ liệu thô và áp dụng các quy tắc làm sạch.

    Quy tắc (theo thứ tự) trong ``clean_data``:

    1. Thêm ``TotalPrice = Quantity × UnitPrice``.
    2. Bỏ hoá đơn huỷ (``InvoiceNo`` bắt đầu bằng ``'C'``).
    3. Chỉ giữ khách hàng ở UK.
    4. Bỏ dòng thiếu ``CustomerID``.
    5. Bỏ dòng có ``Quantity <= 0`` hoặc ``UnitPrice <= 0``.

    Lưu ý: hoá đơn gốc của một đơn đã bị huỷ vẫn được giữ lại, nên doanh thu tính trên
    dữ liệu sạch là doanh thu GỘP (chưa trừ phần huỷ).
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None     # dữ liệu thô (sau load_data)
        self.df_uk: Optional[pd.DataFrame] = None  # dữ liệu sạch (sau clean_data)

    def load_data(self) -> pd.DataFrame:
        """Đọc CSV thô. ``CustomerID`` có kiểu ``Int64`` (số nguyên, cho phép thiếu)."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=RAW_DTYPES,
        )
        self.df["CustomerID"] = self.df["CustomerID"].astype("Int64")

        logger.info("Dataset shape: %s", self.df.shape)
        logger.info("Number of records: %s", f"{len(self.df):,}")
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Áp dụng các quy tắc làm sạch và trả về dữ liệu UK đã làm sạch."""
        if self.df is None:
            raise RuntimeError("Hãy gọi load_data() trước clean_data().")

        df = self.df
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        not_cancelled = ~df["InvoiceNo"].astype(str).str.startswith("C")
        self.df = df[not_cancelled]

        uk = self.df[self.df["Country"] == TARGET_COUNTRY]
        uk = uk.dropna(subset=["CustomerID"])
        uk = uk[(uk["Quantity"] > 0) & (uk["UnitPrice"] > 0)]

        self.df_uk = uk.copy()
        return self.df_uk

    def create_time_features(self) -> None:
        """Thêm ``DayOfWeek`` (0 = thứ Hai) và ``HourOfDay`` vào dữ liệu sạch."""
        if self.df_uk is None:
            raise RuntimeError("Hãy gọi clean_data() trước create_time_features().")
        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour

    def save_cleaned_data(self, output_dir: str = "../data/processed") -> None:
        """Ghi dữ liệu sạch ra ``<output_dir>/cleaned_uk_data.csv``."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "cleaned_uk_data.csv")
        self.df_uk.to_csv(path, index=False)
        logger.info("Cleaned data saved: %s", path)
