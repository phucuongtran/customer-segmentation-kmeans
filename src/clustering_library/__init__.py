# -*- coding: utf-8 -*-
"""Customer Segmentation Library.

Module theo thứ tự pipeline:

- ``cleaner``           : DataCleaner — đọc và làm sạch giao dịch thô        (notebook 01)
- ``rfm``               : compute_rfm — RFM làm mốc so sánh                    (notebook 01)
- ``feature_catalog``   : danh sách 20 đặc trưng, 5 nhóm, FORCE_KEEP          (notebook 02)
- ``customer_features`` : công thức tính từng nhóm đặc trưng                   (notebook 02)
- ``preprocessing``     : quantile → scale → clip, lọc tương quan, trọng số   (notebook 02)
- ``features``          : FeatureEngineer — điều phối các bước trên            (notebook 02)
- ``segmentation``      : tách B2B, K-Means, đánh số cụm ổn định              (notebook 02–05)
- ``profiling``         : KPI kinh doanh và bảng tóm tắt theo cụm              (notebook 05)
- ``visualizer``        : DataVisualizer — biểu đồ EDA                         (notebook 01)
"""

from clustering_library.cleaner import DataCleaner
from clustering_library.features import FeatureEngineer
from clustering_library.rfm import compute_rfm
from clustering_library.segmentation import CLUSTER_NAMES
from clustering_library.visualizer import DataVisualizer
from visual_style import CLUSTER_COLORS

__all__ = [
    "DataCleaner",
    "FeatureEngineer",
    "DataVisualizer",
    "CLUSTER_COLORS",
    "CLUSTER_NAMES",
    "compute_rfm",
]
