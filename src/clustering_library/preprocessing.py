# -*- coding: utf-8 -*-
"""Đưa đặc trưng thô về "model space" cho K-Means.

Chuỗi biến đổi (mỗi bước độc lập theo từng cột):

1. ``QuantileTransformer(output_distribution="normal")`` — ánh xạ theo thứ hạng về
   phân phối chuẩn, nén đuôi dài.
2. ``StandardScaler`` — đưa về trung bình 0, độ lệch chuẩn 1.
3. Cắt về [-CLIP_SIGMA, +CLIP_SIGMA].
4. (chỉ ở bước cuối) Nhân trọng số nhóm ``1 / sqrt(n_đặc_trưng_trong_nhóm)``.

Bộ lọc tương quan chạy trên kết quả bước 1–3 (chưa nhân trọng số), vì đó là không
gian mà khoảng cách Euclid thực sự "nhìn thấy".
"""

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from clustering_library.feature_catalog import FEATURE_TO_FAMILY

logger = logging.getLogger(__name__)

MAX_QUANTILES = 500
CLIP_SIGMA = 3.0
RANDOM_STATE = 42


def fit_quantile_normal(features: pd.DataFrame) -> Tuple[QuantileTransformer, pd.DataFrame]:
    """Fit QuantileTransformer(normal) và trả về (transformer, DataFrame đã biến đổi)."""
    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(MAX_QUANTILES, len(features)),
        random_state=RANDOM_STATE,
    )
    transformed = pd.DataFrame(
        qt.fit_transform(features), columns=features.columns, index=features.index
    )
    return qt, transformed


def fit_standardise_and_clip(features: pd.DataFrame) -> Tuple[StandardScaler, pd.DataFrame]:
    """Fit StandardScaler, rồi cắt về ±CLIP_SIGMA. Trả về (scaler, DataFrame)."""
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(features), columns=features.columns, index=features.index
    )
    return scaler, scaled.clip(lower=-CLIP_SIGMA, upper=CLIP_SIGMA)


def to_model_space(features: pd.DataFrame) -> pd.DataFrame:
    """Quantile → Standardise → Clip, không giữ lại transformer (dùng cho bộ lọc)."""
    _, transformed = fit_quantile_normal(features)
    _, scaled = fit_standardise_and_clip(transformed)
    return scaled


def filter_correlated(
    model_space: pd.DataFrame,
    threshold: float,
    protected: Iterable[str] = (),
) -> List[str]:
    """Bộ lọc tương quan tham lam. Trả về danh sách cột được giữ (giữ nguyên thứ tự).

    1. Bỏ các cột hằng số (trừ cột được bảo vệ).
    2. Lặp: trong các cột có |r| > threshold với ít nhất một cột khác và không được
       bảo vệ, bỏ cột có |r| trung bình cao nhất. Dừng khi không còn cột nào bỏ được.
    """
    cols = list(model_space.columns)
    protected = set(protected) & set(cols)

    constant = [c for c in cols if model_space[c].std() == 0 and c not in protected]
    if constant:
        logger.info("Bỏ %d đặc trưng hằng số: %s", len(constant), constant)
        cols = [c for c in cols if c not in constant]
    if protected:
        logger.info("Luôn giữ: %s", sorted(protected))

    logger.info("Bộ lọc tương quan (model space) — bắt đầu với %d ứng viên:", len(cols))
    while True:
        abs_corr = np.nan_to_num(model_space[cols].corr().abs().to_numpy(), nan=0.0)
        np.fill_diagonal(abs_corr, 0.0)

        violating = (abs_corr > threshold).any(axis=1)
        droppable = [c for c, v in zip(cols, violating) if v and c not in protected]
        if not droppable:
            if violating.any():
                logger.info("Các vi phạm còn lại đều giữa đặc trưng được bảo vệ — dừng.")
            break

        mean_corr = pd.Series(abs_corr.mean(axis=0), index=cols)
        drop = mean_corr[droppable].idxmax()
        i = cols.index(drop)
        partner = int(abs_corr[i].argmax())
        logger.info(
            "Bỏ '%s' (|r|=%.3f với '%s', mean|r|=%.3f)",
            drop, abs_corr[i, partner], cols[partner], mean_corr[drop],
        )
        cols.remove(drop)

    logger.info("Giữ lại %d đặc trưng.", len(cols))
    return cols


def family_weights(features: Iterable[str]) -> Dict[str, float]:
    """Trọng số 1/sqrt(n) cho mỗi đặc trưng, n = số đặc trưng được giữ trong cùng nhóm.

    Nhờ đó mỗi nhóm hành vi có tổng bình phương trọng số bằng 1 trong khoảng cách Euclid,
    nhóm nhiều đặc trưng không lấn át nhóm ít đặc trưng.
    """
    features = list(features)
    family_sizes = pd.Series([FEATURE_TO_FAMILY[f] for f in features]).value_counts()
    return {f: float(1.0 / np.sqrt(family_sizes[FEATURE_TO_FAMILY[f]])) for f in features}
