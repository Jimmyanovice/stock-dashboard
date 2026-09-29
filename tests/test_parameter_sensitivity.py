# -*- coding: utf-8 -*-
"""
参数敏感性分析与样本时序切分测试套件 (tests/test_parameter_sensitivity.py)
---------------------------------------------------------------------
验证针对缺陷 B11 / 任务 T7 的修复:
- 训练集 (2019-2023)、验证集 (2024)、样本外测试集 (2025-2026) 时序因果切分
- 核心超参数 ±20% 敏感性压力测试与稳健性评估
- sample_split_performance.csv 与 param_sensitivity.csv 数据表完整性与规范性
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.hedging_strategies import HedgingEngine
from src.models.parameter_sensitivity import (
    BASELINE_PARAMS,
    split_dataset,
    evaluate_sample_splits,
    run_parameter_sensitivity
)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "czce_sa_fg_aligned_features.csv"
TABLES_DIR = ROOT / "reports" / "tables"


@pytest.fixture(scope="module")
def full_features_df() -> pd.DataFrame:
    """加载已对齐清洗的特征全量数据"""
    assert FEATURES_CSV.exists(), f"缺失特征工程数据文件: {FEATURES_CSV}"
    return pd.read_csv(FEATURES_CSV)


def test_sample_split_dataset_causality_and_integrity(full_features_df: pd.DataFrame):
    """
    【T7 验收标准】
    检验样本时序切分的因果性与样本完整性:
    1. 各切片时间区间严格单调递增且互不重叠
    2. 训练集 (988行) + 验证集 (242行) + 测试集 (422行) == 全样本 (1652行)
    """
    splits = split_dataset(full_features_df)

    train_df = splits["训练期 (2019-2023)"]
    val_df = splits["验证期 (2024)"]
    test_df = splits["样本外测试期 (2025-2026)"]
    full_df = splits["全历史样本 (2019-2026)"]

    assert len(train_df) == 988, f"训练期样本数异常: {len(train_df)} != 988"
    assert len(val_df) == 242, f"验证期样本数异常: {len(val_df)} != 242"
    assert len(test_df) == 422, f"样本外测试期样本数异常: {len(test_df)} != 422"
    assert len(train_df) + len(val_df) + len(test_df) == len(full_df) == 1652

    # 时间序列连续无重叠校验
    assert train_df["date"].max() <= "2023-12-31"
    assert val_df["date"].min() >= "2024-01-01"
    assert val_df["date"].max() <= "2024-12-31"
    assert test_df["date"].min() >= "2025-01-01"


def test_hedging_engine_accepts_custom_configs(full_features_df: pd.DataFrame):
    """
    验证 HedgingEngine.run_trend_gate_collar 能够正确接收参数字典并生效。
    """
    sample_df = full_features_df.iloc[:100].copy()
    engine = HedgingEngine(sample_df)

    # 1. 默认参数运行
    res_default = engine.run_trend_gate_collar()
    assert "net_worth" in res_default.columns
    assert "trend_gate" in res_default.columns

    # 2. 传入自定义参数
    res_custom = engine.run_trend_gate_collar(
        trend_gate_config={"fast_span": 10, "slow_span": 30, "threshold": 0.40},
        nale_config={"sa_to_fg_lag": 14.4}
    )
    assert len(res_custom) == len(sample_df)


def test_sample_split_performance_table_generation(full_features_df: pd.DataFrame, tmp_path: Path):
    """
    验证时序切片回测对比表能够成功生成，并包含完整指标列。
    """
    test_csv = tmp_path / "test_split_perf.csv"
    res_df = evaluate_sample_splits(full_features_df, output_path=test_csv)

    assert test_csv.exists()
    assert not res_df.empty

    required_cols = [
        "样本阶段", "起始日期", "结束日期", "交易日数", "方案名称",
        "最终资产净值", "累计收益率(%)", "套保有效性HE(%)", "最大回撤(%)",
        "年化波动率(%)", "平均保证金占用(万元)", "峰值保证金占用(万元)"
    ]
    for col in required_cols:
        assert col in res_df.columns, f"字段缺失: {col}"

    # 验证四大方案均涵盖
    schemes = res_df["方案名称"].unique().tolist()
    assert any("方案A" in s for s in schemes)
    assert any("方案B" in s for s in schemes)
    assert any("方案C" in s for s in schemes)
    assert any("方案D" in s for s in schemes)


def test_param_sensitivity_table_generation(full_features_df: pd.DataFrame, tmp_path: Path):
    """
    【B11 核心验收标准】
    验证 8 项核心参数的 ±20% 敏感性压力测试表完整生成。
    """
    test_csv = tmp_path / "test_param_sens.csv"
    sens_df = run_parameter_sensitivity(full_features_df, output_path=test_csv)

    assert test_csv.exists()
    assert not sens_df.empty

    # 验证包含 8 项核心参数
    tested_params = sens_df["参数名称"].unique().tolist()
    for param_name in BASELINE_PARAMS.keys():
        assert param_name in tested_params, f"核心参数 {param_name} 未在敏感性表中测试！"

    # 验证每项参数均包含 -20%, 0%, +20% 扰动
    for param_name in BASELINE_PARAMS.keys():
        sub = sens_df[sens_df["参数名称"] == param_name]
        assert len(sub) == 3, f"参数 {param_name} 扰动测试行数应为 3，实为 {len(sub)}"

    # 验证稳健性评级有效
    assert set(sens_df["稳健性评级"].unique()).issubset({"ROBUST", "SENSITIVE"})
