"""概念漂移监控（S12.8，模型治理）。

模型上线后，线上数据分布会随时间偏离训练时的分布（概念漂移/协变量漂移），导致评分
逐渐失真。本模块在「训练时」记录各特征的分布基线，在「评估新数据时」用 **PSI
（Population Stability Index，群体稳定性指标）** 度量新旧分布偏移，超阈值则告警并建议重训。

PSI 解释（行业惯例阈值）：
- PSI < 0.1：分布稳定，无需动作。
- 0.1 ≤ PSI < 0.25：轻微漂移，需关注。
- PSI ≥ 0.25：显著漂移，建议重训模型。

实现为纯 Python、确定性（固定分箱边界由基线分位数确定），无第三方依赖。

DDD 分层：本模块属 domain 层，仅依赖标准库与领域内特征抽取（fraud_model.extract_features）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from app.domain.fraud_model import extract_features

# PSI 漂移判定阈值（行业惯例）
PSI_STABLE_THRESHOLD = 0.1
PSI_SHIFT_THRESHOLD = 0.25
# 分箱数（基线按分位数分箱）
_DEFAULT_BINS = 10
# 拉普拉斯平滑，避免某箱占比为 0 时 PSI 出现 inf
_EPS = 1e-6


@dataclass
class FeatureBaseline:
    """单特征的分布基线：分箱边界 + 各箱基线占比。"""

    bin_edges: list[float]
    base_ratios: list[float]


@dataclass
class DriftBaseline:
    """训练时记录的整体特征分布基线（供后续漂移检测对比）。"""

    feature_baselines: dict[str, FeatureBaseline] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            name: {"binEdges": fb.bin_edges, "baseRatios": fb.base_ratios}
            for name, fb in self.feature_baselines.items()
        }

    @staticmethod
    def from_dict(data: dict) -> "DriftBaseline":
        fbs: dict[str, FeatureBaseline] = {}
        for name, fb in (data or {}).items():
            fbs[name] = FeatureBaseline(
                bin_edges=list(fb.get("binEdges", [])),
                base_ratios=list(fb.get("baseRatios", [])),
            )
        return DriftBaseline(feature_baselines=fbs)


def build_drift_baseline(
    samples: Sequence[object], *, bins: int = _DEFAULT_BINS
) -> DriftBaseline:
    """从训练样本构建特征分布基线（按分位数分箱，记录各箱占比）。确定性。"""
    feature_rows = [extract_features(r) for r in samples]
    feature_rows = [f for f in feature_rows if f]
    columns = sorted({name for f in feature_rows for name in f})

    baselines: dict[str, FeatureBaseline] = {}
    for col in columns:
        values = sorted(float(f.get(col, 0.0)) for f in feature_rows)
        edges = _quantile_edges(values, bins)
        ratios = _bin_ratios(values, edges)
        baselines[col] = FeatureBaseline(bin_edges=edges, base_ratios=ratios)
    return DriftBaseline(feature_baselines=baselines)


def compute_drift(
    baseline: DriftBaseline, samples: Sequence[object]
) -> dict:
    """对新样本计算各特征 PSI 与整体漂移结论（确定性）。

    返回：{
      "featurePsi": {特征: psi},
      "maxPsi": 最大 PSI,
      "drifted": bool（maxPsi ≥ 0.25）,
      "level": "stable" | "minor" | "shift",
      "recommendation": 中文建议,
    }
    """
    feature_rows = [extract_features(r) for r in samples]
    feature_rows = [f for f in feature_rows if f]

    feature_psi: dict[str, float] = {}
    for col, fb in baseline.feature_baselines.items():
        values = [float(f.get(col, 0.0)) for f in feature_rows]
        new_ratios = _bin_ratios(sorted(values), fb.bin_edges)
        feature_psi[col] = round(_psi(fb.base_ratios, new_ratios), 6)

    max_psi = max(feature_psi.values()) if feature_psi else 0.0
    if max_psi >= PSI_SHIFT_THRESHOLD:
        level = "shift"
        recommendation = "检测到显著分布漂移（PSI≥0.25），建议尽快用近期数据重训模型"
    elif max_psi >= PSI_STABLE_THRESHOLD:
        level = "minor"
        recommendation = "检测到轻微分布漂移（0.1≤PSI<0.25），建议持续关注并准备重训"
    else:
        level = "stable"
        recommendation = "分布稳定（PSI<0.1），无需动作"

    return {
        "featurePsi": dict(sorted(feature_psi.items(), key=lambda kv: kv[1], reverse=True)),
        "maxPsi": round(max_psi, 6),
        "drifted": max_psi >= PSI_SHIFT_THRESHOLD,
        "level": level,
        "recommendation": recommendation,
    }


# ---- 内部工具 ------------------------------------------------------------------


def _quantile_edges(sorted_values: list[float], bins: int) -> list[float]:
    """按分位数生成 bins+1 个分箱边界（去重；全相同值时退化为单箱）。"""
    if not sorted_values:
        return [0.0, 1.0]
    lo, hi = sorted_values[0], sorted_values[-1]
    if hi <= lo:
        # 常量列：构造一个能容纳该值的单箱
        return [lo - 0.5, lo + 0.5]
    edges = []
    n = len(sorted_values)
    for b in range(bins + 1):
        idx = int(round(b / bins * (n - 1)))
        edges.append(sorted_values[idx])
    # 去重并保证严格递增（边界并列会导致空箱）
    uniq: list[float] = []
    for e in edges:
        if not uniq or e > uniq[-1]:
            uniq.append(e)
    if len(uniq) < 2:
        uniq = [lo, hi]
    # 末边界略微抬高，确保最大值落入最后一箱
    uniq[-1] = uniq[-1] + abs(uniq[-1]) * 1e-9 + 1e-9
    return uniq


def _bin_ratios(sorted_values: list[float], edges: list[float]) -> list[float]:
    """统计各箱占比（左闭右开，最后一箱右闭），返回长度 len(edges)-1 的占比列表。"""
    n_bins = max(1, len(edges) - 1)
    counts = [0] * n_bins
    total = len(sorted_values)
    if total == 0:
        return [0.0] * n_bins
    for v in sorted_values:
        b = _bin_index(v, edges)
        counts[b] += 1
    return [c / total for c in counts]


def _bin_index(value: float, edges: list[float]) -> int:
    """定位 value 所属箱序号（钳制到 [0, n_bins-1]）。"""
    n_bins = max(1, len(edges) - 1)
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return n_bins - 1
    # 线性扫描（箱数小，无需二分）
    for b in range(n_bins):
        if edges[b] <= value < edges[b + 1]:
            return b
    return n_bins - 1


def _psi(base_ratios: list[float], new_ratios: list[float]) -> float:
    """计算 PSI：Σ (new - base) * ln(new / base)，带拉普拉斯平滑避免除零。"""
    psi = 0.0
    for base, new in zip(base_ratios, new_ratios):
        b = base + _EPS
        nw = new + _EPS
        psi += (nw - b) * math.log(nw / b)
    return psi
