# -*- coding: utf-8 -*-
"""Pipeline đặc trưng cấp khách hàng cho K-Means.

    20 ứng viên (``customer_features``)
      → model space: QuantileTransformer(normal) → StandardScaler → cắt ±3σ (``preprocessing``)
      → bộ lọc tương quan |r| > 0.85, giữ các đặc trưng trong FORCE_KEEP
      → fit lại transform trên các đặc trưng được giữ
      → nhân trọng số nhóm 1/sqrt(n) để 5 nhóm hành vi có tổng trọng số bằng nhau

Bộ lọc chạy SAU biến đổi quantile là có chủ đích: biến đổi đơn điệu giữ thứ hạng nhưng
có thể biến hai đặc trưng tương quan yếu ở dạng thô thành gần trùng nhau trong model space.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from clustering_library import feature_catalog, preprocessing
from clustering_library.customer_features import compute_all_candidates

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Tính, lọc và chuẩn hoá đặc trưng hành vi khách hàng.

    Ví dụ::

        fe = FeatureEngineer(
            data_path="data/processed/cleaned_uk_data.csv",
            raw_data_path="data/raw/online_retail.csv",
        )
        fe.load_data()
        fe.create_customer_features()   # 20 ứng viên → lọc tương quan → 14 đặc trưng
        fe.transform_features()         # QuantileTransformer(normal)
        fe.scale_features()             # StandardScaler + cắt ±3σ + trọng số nhóm
        fe.save_pipeline("data/models")

    Chỉ cần đặc trưng thô (không lọc, không chuẩn hoá)? Gọi ``fe.compute_candidates()``.
    """

    # Giữ làm thuộc tính lớp để code cũ (``FeatureEngineer.CANDIDATES``...) vẫn chạy.
    CANDIDATES: List[str] = feature_catalog.CANDIDATES
    FEATURE_FAMILIES: Dict[str, List[str]] = feature_catalog.FEATURE_FAMILIES
    FORCE_KEEP: List[str] = feature_catalog.FORCE_KEEP
    BULK_QTY_THRESHOLD: int = feature_catalog.BULK_QTY_THRESHOLD

    def __init__(
        self,
        data_path: str,
        raw_data_path: Optional[str] = None,
        corr_threshold: float = 0.85,
        force_keep: Optional[List[str]] = None,
    ) -> None:
        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.corr_threshold = corr_threshold
        self.force_keep: List[str] = self.FORCE_KEEP if force_keep is None else force_keep

        # Dữ liệu đầu vào
        self.df: Optional[pd.DataFrame] = None       # giao dịch đã làm sạch
        self.raw_df: Optional[pd.DataFrame] = None   # giao dịch thô (để tính ReturnRate)

        # Kết quả từng bước
        self.customer_feature_candidates: Optional[pd.DataFrame] = None  # 20 cột thô
        self.feature_customer: List[str] = []                            # tên cột được giữ
        self.customer_features: Optional[pd.DataFrame] = None            # cột được giữ, thô
        self.customer_features_transformed: Optional[pd.DataFrame] = None
        self.customer_features_scaled_unweighted: Optional[pd.DataFrame] = None
        self.customer_features_scaled: Optional[pd.DataFrame] = None     # đầu vào K-Means
        self.family_weights: Dict[str, float] = {}

        # Transformer đã fit (để lưu và áp dụng cho khách mới)
        self.qt: Optional[QuantileTransformer] = None
        self.scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------ #
    # Tải dữ liệu                                                          #
    # ------------------------------------------------------------------ #

    def load_data(self) -> pd.DataFrame:
        """Đọc giao dịch đã làm sạch (bắt buộc) và giao dịch thô (nếu có)."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])
        self.df["CustomerID"] = self.df["CustomerID"].astype("int64")
        logger.info("Cleaned dataset shape: %s", self.df.shape)

        if self.raw_data_path and os.path.exists(self.raw_data_path):
            self.raw_df = pd.read_csv(self.raw_data_path, encoding="ISO-8859-1", low_memory=False)
            self.raw_df["InvoiceDate"] = pd.to_datetime(self.raw_df["InvoiceDate"])
            logger.info("Raw dataset shape: %s", self.raw_df.shape)
        elif self.raw_data_path:
            logger.warning("Không tìm thấy raw_data_path (%s). ReturnRate sẽ bằng 0.", self.raw_data_path)
        return self.df

    # ------------------------------------------------------------------ #
    # Pipeline                                                             #
    # ------------------------------------------------------------------ #

    def compute_candidates(self) -> pd.DataFrame:
        """Tính 20 đặc trưng ứng viên ở đơn vị gốc (index = CustomerID)."""
        self._require(self.df, "load_data()")
        self.customer_feature_candidates = compute_all_candidates(
            self.df, self.raw_df, self.BULK_QTY_THRESHOLD
        )
        return self.customer_feature_candidates

    def create_customer_features(self) -> pd.DataFrame:
        """Tính ứng viên rồi giữ lại tập đặc trưng không trùng lặp (lọc trong model space).

        Trả về DataFrame các đặc trưng được giữ ở đơn vị gốc, có cột ``CustomerID``.
        """
        candidates = self.compute_candidates()
        self.feature_customer = preprocessing.filter_correlated(
            preprocessing.to_model_space(candidates),
            threshold=self.corr_threshold,
            protected=self.force_keep,
        )
        self.customer_features = candidates[self.feature_customer].reset_index()

        logger.info("Tập đặc trưng cuối (%d):", len(self.feature_customer))
        for name in self.feature_customer:
            logger.info("    %s", name)
        return self.customer_features

    def transform_features(self) -> pd.DataFrame:
        """Fit QuantileTransformer(normal) trên các đặc trưng được giữ."""
        self._require(self.customer_features, "create_customer_features()")
        selected = self.customer_features.set_index("CustomerID")
        self.qt, self.customer_features_transformed = preprocessing.fit_quantile_normal(selected)
        logger.info("QuantileTransformer(normal) applied.")
        return self.customer_features_transformed

    def scale_features(self) -> pd.DataFrame:
        """StandardScaler → cắt ±3σ → nhân trọng số nhóm. Kết quả là đầu vào K-Means."""
        self._require(self.customer_features_transformed, "transform_features()")
        self.scaler, clipped = preprocessing.fit_standardise_and_clip(
            self.customer_features_transformed
        )
        self.customer_features_scaled_unweighted = clipped
        self.family_weights = preprocessing.family_weights(self.feature_customer)
        self.customer_features_scaled = clipped.mul(pd.Series(self.family_weights), axis=1)
        logger.info("Feature scaling complete (±3σ clip + family balancing applied).")
        return self.customer_features_scaled

    # ------------------------------------------------------------------ #
    # Lưu kết quả                                                          #
    # ------------------------------------------------------------------ #

    def save_pipeline(self, output_dir: str = "../models") -> None:
        """Lưu scaler.pkl, quantile_transformer.pkl và pipeline_metadata.json."""
        self._require(self.qt, "transform_features()")
        self._require(self.scaler, "scale_features()")

        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.qt, os.path.join(output_dir, "quantile_transformer.pkl"))

        metadata = {
            "bulk_qty_threshold": self.BULK_QTY_THRESHOLD,
            "feature_order": self.feature_customer,
            "n_candidates": len(self.CANDIDATES),
            "corr_threshold": self.corr_threshold,
            "correlation_space": "quantile_transformed_scaled_clipped",
            "post_scale_clip": preprocessing.CLIP_SIGMA,
            "family_weights": self.family_weights,
            "feature_families": self.FEATURE_FAMILIES,
        }
        with open(os.path.join(output_dir, "pipeline_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Pipeline artifacts saved to: %s/", output_dir)

    def save_features(self, output_dir: str = "../data/processed") -> None:
        """Lưu mọi bảng đặc trưng trung gian, và pipeline vào thư mục ``models`` cạnh đó."""
        self._require(self.customer_features_scaled, "scale_features()")
        os.makedirs(output_dir, exist_ok=True)

        tables = {
            "customer_feature_candidates": self.customer_feature_candidates,
            "customer_features": self.customer_features.set_index("CustomerID"),
            "customer_features_transformed": self.customer_features_transformed,
            "customer_features_scaled_unweighted": self.customer_features_scaled_unweighted,
            "customer_features_scaled": self.customer_features_scaled,
        }
        for name, table in tables.items():
            table.to_csv(os.path.join(output_dir, f"{name}.csv"))
        logger.info("All features saved to: %s", output_dir)

        self.save_pipeline(os.path.join(os.path.dirname(output_dir), "models"))

    # ------------------------------------------------------------------ #

    @staticmethod
    def _require(value: object, step: str) -> None:
        if value is None:
            raise RuntimeError(f"Chưa có dữ liệu cho bước này. Hãy chạy {step} trước.")
