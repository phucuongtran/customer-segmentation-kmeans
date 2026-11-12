# -*- coding: utf-8 -*-
"""Single source of truth for the behavioural feature set.

Mọi nơi cần biết "có những đặc trưng nào, thuộc nhóm nào" đều import từ đây:
``customer_features`` (tính toán), ``preprocessing`` (trọng số nhóm),
``FeatureEngineer`` (pipeline) và tests.

Thứ tự trong ``CANDIDATES`` là thứ tự cột của ma trận đặc trưng. Đừng đổi thứ tự:
bộ lọc tương quan duyệt cột theo thứ tự này nên kết quả có thể thay đổi.
"""

from typing import Dict, List

# 20 đặc trưng ứng viên, nhóm theo FEATURE_FAMILIES (xem bên dưới).
CANDIDATES: List[str] = [
    # Purchase Rhythm
    "Recency",               # số ngày từ lần mua cuối tới snapshot (max date + 1 ngày)
    "InterPurchaseCV",       # CV của khoảng cách (ngày) giữa các hoá đơn liên tiếp
    "ActiveSpanDays",        # ngày mua cuối − ngày mua đầu
    "MonthlyOrderRate",      # số hoá đơn / max(ActiveSpanDays / 30, 1)
    # Spending Shape
    "AOV",                   # tổng chi tiêu / số hoá đơn
    "SpendGini",             # hệ số Gini của giá trị các hoá đơn
    "PriceCV",               # CV của UnitPrice trên từng dòng
    # Basket Behaviour
    "NewSKURate",            # trung bình tỉ lệ SKU chưa từng mua ở mỗi hoá đơn
    "RepeatSKUFraction",     # tỉ lệ SKU xuất hiện ở > 1 hoá đơn
    "BasketSizeCV",          # CV của số SKU phân biệt mỗi hoá đơn
    # Volume & Bulk
    "BurstIndex",            # giá trị hoá đơn lớn nhất / giá trị hoá đơn trung bình
    "BulkLineRate",          # tỉ lệ dòng có Quantity >= BULK_QTY_THRESHOLD
    "QuantityCV",            # CV của Quantity trên từng dòng
    "MedianBasketQty",       # trung vị tổng Quantity mỗi hoá đơn
    "SKU_HHI",               # Σ(tỉ trọng chi tiêu theo SKU)² — độ tập trung sản phẩm
    # Spending Shape (tiếp)
    "MaxSpendShare",         # hoá đơn lớn nhất / tổng chi tiêu
    # Customer Lifecycle
    "ReturnRate",            # số hoá đơn huỷ (C-prefix) / tổng số hoá đơn (dữ liệu thô)
    "ReturnValueRate",       # |giá trị huỷ| / |tổng giá trị| (dữ liệu thô)
    # Spending Shape (tiếp)
    "SpendAcceleration",     # TB 3 tháng có mua cuối / TB 3 tháng có mua đầu − 1, cắt [-3, 3]
    # Customer Lifecycle (tiếp)
    "QuarterConcentration",  # chi tiêu của quý lớn nhất / tổng chi tiêu, trong [0.25, 1]
]

# 5 nhóm hành vi. Trọng số mỗi đặc trưng = 1 / sqrt(số đặc trưng còn lại trong nhóm)
# để mỗi nhóm đóng góp tổng bình phương trọng số bằng nhau vào khoảng cách Euclid.
FEATURE_FAMILIES: Dict[str, List[str]] = {
    "Purchase Rhythm": [
        "Recency", "InterPurchaseCV", "ActiveSpanDays", "MonthlyOrderRate",
    ],
    "Spending Shape": [
        "AOV", "SpendGini", "PriceCV", "MaxSpendShare", "SpendAcceleration",
    ],
    "Basket Behaviour": [
        "NewSKURate", "RepeatSKUFraction", "BasketSizeCV",
    ],
    "Volume & Bulk": [
        "BurstIndex", "BulkLineRate", "QuantityCV", "MedianBasketQty", "SKU_HHI",
    ],
    "Customer Lifecycle": [
        "ReturnRate", "ReturnValueRate", "QuarterConcentration",
    ],
}

# Mỗi trục kinh doanh chính giữ một "mỏ neo" dễ diễn giải; bộ lọc tương quan không
# bao giờ loại các đặc trưng này. ReturnValueRate cố ý không được bảo vệ: trong
# model space nó gần như trùng ReturnRate và sẽ nhân đôi trọng số hành vi huỷ đơn.
FORCE_KEEP: List[str] = [
    "Recency", "MonthlyOrderRate", "ReturnRate", "QuarterConcentration", "SKU_HHI",
]

# Một dòng hàng được coi là "mua sỉ" khi Quantity >= ngưỡng này.
BULK_QTY_THRESHOLD: int = 12

FEATURE_TO_FAMILY: Dict[str, str] = {
    feature: family
    for family, features in FEATURE_FAMILIES.items()
    for feature in features
}

assert set(FEATURE_TO_FAMILY) == set(CANDIDATES), "Mỗi ứng viên phải thuộc đúng một nhóm"
assert len(FEATURE_TO_FAMILY) == len(CANDIDATES), "Một đặc trưng xuất hiện ở hai nhóm"
