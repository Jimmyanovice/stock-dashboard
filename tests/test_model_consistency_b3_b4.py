# -*- coding: utf-8 -*-
"""
模型一致性专项测试套件 (tests/test_model_consistency_b3_b4.py)
------------------------------------------------------------
验证针对缺陷 B3 与 B4 的修复:
- B3: Trend Gate 统一打分权重表与缺失信号中性等价性 (tsmom=None vs 0.0)
- B4: Crush Spread 跨特征工程与 NALE 内部的单一定义与燃料扣除口径一致性
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    calculate_crush_spread,
    compute_commodity_spreads,
    DEFAULT_FUEL_COST_PER_TON,
    DEFAULT_SA_STOICHIOMETRIC_RATIO
)
from src.models.temporal_nale import TemporalNALE, TemporalNALEConfig
from src.models.trend_gate import TrendGateMachine

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "processed" / "czce_sa_fg_aligned_features.csv"


# =========================================================================
# 1. 缺陷 B3 专项验证: Trend Gate 统一权重与中性填补
# =========================================================================

def test_b3_tsmom_none_vs_zero_strict_numerical_identity():
    """
    【B3 核心验收标准】
    传入 tsmom=None 与传入 tsmom=0.0，其 composite_score 必须在每一采样点上严格恒等 (误差 == 0.0)。
    消除由于条件分支导致的同数据买卖信号相反的重大隐患。
    """
    np.random.seed(2026)
    n = 200
    prices = pd.Series(1500.0 + np.cumsum(np.random.normal(0, 8, n)))
    nale = pd.Series(np.random.uniform(-1.0, 1.0, n))

    gate = TrendGateMachine()

    # 1. 传入 None
    res_none = gate.evaluate_regime(prices, nale_signals=nale, tsmom_signals=None)

    # 2. 传入全 0.0 序列
    res_zero = gate.evaluate_regime(prices, nale_signals=nale, tsmom_signals=pd.Series(0.0, index=prices.index))

    # 逐点比较
    diff_score = np.abs(res_none["composite_score"].values - res_zero["composite_score"].values)
    assert np.max(diff_score) == 0.0, f"composite_score 存在数值差异: 最大偏差 {np.max(diff_score)}"

    diff_gate = np.abs(res_none["trend_gate"].values - res_zero["trend_gate"].values)
    assert np.max(diff_gate) == 0.0, "状态机 trend_gate 状态序列不一致！"

    diff_hedge = np.abs(res_none["dynamic_hedge_ratio"].values - res_zero["dynamic_hedge_ratio"].values)
    assert np.max(diff_hedge) == 0.0, "对冲比率 dynamic_hedge_ratio 不一致！"


def test_b3_missing_signals_neutral_fill_preserves_denominator():
    """
    【B3 架构规范】
    缺失信号按中性 0.0 填补，严格不允许改变分母导致单因子权重漂移。
    """
    gate = TrendGateMachine()
    assert gate.price_weight == pytest.approx(0.45, rel=1e-6)
    assert gate.nale_weight == pytest.approx(0.35, rel=1e-6)
    assert gate.tsmom_weight == pytest.approx(0.20, rel=1e-6)

    # 当处于完全均线破位时 (trend_score = 1.0)
    # 若 nale 与 tsmom 缺失，得分严格为 0.45 * 1.0 + 0.35 * 0.0 + 0.20 * 0.0 = 0.45
    prices = pd.Series([100.0] * 50 + [100.0 - i * 2.0 for i in range(1, 50)])
    res = gate.evaluate_regime(prices)

    last_score = res["composite_score"].iloc[-1]
    assert last_score == pytest.approx(0.45, rel=1e-6), f"期望综合得分为 0.45，实测为 {last_score}"


def test_b3_active_defense_triggers_only_on_deep_backwardation():
    """
    测试深贴水主动防御机制仅在 TSMOM < -0.2 时激活抑制，中性 0.0 信号不激活抑制。
    """
    prices = pd.Series([1500.0] * 30 + [1450.0] * 30)
    gate = TrendGateMachine(active_defense=True)

    # 1. 中性 TSMOM = 0.0 (未激活抑制)
    res_neutral = gate.evaluate_regime(prices, tsmom_signals=pd.Series(0.0, index=prices.index))

    # 2. 深度贴水 TSMOM = -0.8 (激活抑制，suppression_factor = 1.0 + 0.5 * (-0.8) = 0.6)
    res_defended = gate.evaluate_regime(prices, tsmom_signals=pd.Series(-0.8, index=prices.index))

    score_neutral = res_neutral["composite_score"].iloc[-1]
    score_defended = res_defended["composite_score"].iloc[-1]

    if score_neutral > 0.0:
        assert score_defended < score_neutral, "主动防御未有效抑制破位得分！"
        expected_ratio = 1.0 + 0.5 * (-0.8)  # 0.6
        assert (score_defended / score_neutral) == pytest.approx(expected_ratio, rel=1e-5)


# =========================================================================
# 2. 缺陷 B4 专项验证: Crush Spread 口径与跨模块一致性
# =========================================================================

def test_b4_calculate_crush_spread_formula():
    """
    【B4 核心验收标准】
    虚拟压榨利润计算公式严格满足: CrushSpread = P_fg - 0.20 * P_sa - 350.0
    """
    p_fg = 1500.0
    p_sa = 1800.0
    expected = 1500.0 - 0.20 * 1800.0 - 350.0  # 1500 - 360 - 350 = 790.0

    # 标量计算
    res_scalar = calculate_crush_spread(p_fg, p_sa)
    assert res_scalar == pytest.approx(expected, rel=1e-6)

    # Series 计算
    s_fg = pd.Series([1500.0, 1600.0])
    s_sa = pd.Series([1800.0, 2000.0])
    s_res = calculate_crush_spread(s_fg, s_sa)
    assert s_res.iloc[0] == pytest.approx(expected, rel=1e-6)
    assert s_res.iloc[1] == pytest.approx(1600.0 - 0.20 * 2000.0 - 350.0, rel=1e-6)


def test_b4_feature_engineering_and_temporal_nale_function_reuse():
    """
    【B4 架构一致性】
    TemporalNALE 模块与 feature_engineering 共享同一函数 calculate_crush_spread。
    """
    p_fg = np.array([1200.0, 1350.0, 1500.0])
    p_sa = np.array([1600.0, 1700.0, 1800.0])

    res_fe = calculate_crush_spread(p_fg, p_sa, sa_stoichiometric_ratio=0.20, fuel_cost=350.0)

    nale = TemporalNALE()
    res_nale = nale.compute_virtual_crush_spread(p_fg, p_sa, sa_stoichiometric_ratio=0.20, fuel_cost_per_ton=350.0)

    np.testing.assert_allclose(res_fe, res_nale, rtol=1e-9)


def test_b4_processed_csv_crush_spread_identity():
    """
    【B4 全历史数据核验】
    验证 data/processed/czce_sa_fg_aligned_features.csv 中全部 1652 行数据
    满足: crush_spread == close_fg - 0.20 * close_sa - 350.0
    """
    assert FEATURES_CSV.exists(), f"未找到特征数据文件: {FEATURES_CSV}"
    df = pd.read_csv(FEATURES_CSV)

    assert "crush_spread" in df.columns
    assert "close_fg" in df.columns
    assert "close_sa" in df.columns

    expected = df["close_fg"] - 0.20 * df["close_sa"] - 350.0
    max_abs_diff = (df["crush_spread"] - expected).abs().max()

    assert max_abs_diff < 1e-4, f"全历史 crush_spread 存在偏离，最大误差: {max_abs_diff}"


def test_b4_temporal_nale_output_identical_to_pipeline_column():
    """
    【B4 模型内外部一致性】
    当将特征工程 DataFrame 送入 TemporalNALE 时，其内部复用的 spread 与 df['crush_spread']
    以及全新通过 calculate_crush_spread 算出的值完全一致。
    """
    df = pd.read_csv(FEATURES_CSV).iloc[:100].copy()
    nale = TemporalNALE()
    res = nale.compute_chain_response(df)

    expected_crush = calculate_crush_spread(df["close_fg"], df["close_sa"], 0.20, 350.0)

    np.testing.assert_allclose(res["crush_spread"].values, df["crush_spread"].values, rtol=1e-5)
    np.testing.assert_allclose(res["crush_spread"].values, expected_crush.values, rtol=1e-5)
