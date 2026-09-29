# -*- coding: utf-8 -*-
"""
Trend Gate 因果门控状态机测试套件 (test_trend_gate.py)
"""

import numpy as np
import pandas as pd
import pytest

from src.models.trend_gate import TrendGateMachine

def test_trend_gate_basic_breakdown():
    """测试在单边持续下跌序列中状态机准确触发 1.0 主跌浪状态 (纯技术面权重)"""
    # 构造先平稳后单边暴跌序列
    prices = pd.Series([100.0] * 50 + [100.0 - i * 1.5 for i in range(1, 60)])
    machine = TrendGateMachine(
        fast_span=10,
        slow_span=30,
        hysteresis_window=3,
        price_weight=1.0,
        nale_weight=0.0,
        tsmom_weight=0.0
    )
    res = machine.evaluate_regime(prices)
    
    assert res["trend_gate"].iloc[20] == 0.0, "初始平稳期门控状态必须为 0"
    assert res["trend_gate"].iloc[-1] == 1.0, "持续深跌后门控状态必须阶跃为 1"
    assert res["dynamic_hedge_ratio"].iloc[-1] == 0.95, "破位状态下对冲比率应为 0.95"


def test_trend_gate_causal_property():
    """因果性测试：更改未来行情，历史所有门控判定和对冲比率必须完全一致"""
    np.random.seed(42)
    p_orig = pd.Series(np.cumsum(np.random.normal(0, 1, 120)) + 1000.0)
    p_mod = p_orig.copy()
    
    split_point = 80
    p_mod.iloc[split_point:] += 300.0  # 突发未来暴涨
    
    machine = TrendGateMachine()
    res_orig = machine.evaluate_regime(p_orig)
    res_mod = machine.evaluate_regime(p_mod)
    
    pd.testing.assert_series_equal(
        res_orig["trend_gate"].iloc[:split_point],
        res_mod["trend_gate"].iloc[:split_point],
        check_names=False
    )


def test_trend_gate_with_nale_integration():
    """测试结合 Temporal NALE 先行信号后的门控响应"""
    prices = pd.Series([2000.0] * 80)
    # 模拟上游传导的强利润挤压预警信号
    nale_signals = pd.Series([0.0] * 40 + [0.8] * 40)
    
    machine = TrendGateMachine(nale_weight=0.5)
    res = machine.evaluate_regime(prices, nale_signals=nale_signals)
    
    assert "composite_score" in res.columns
    assert (res["composite_score"].iloc[50:] > res["composite_score"].iloc[10]).all()


def test_trend_gate_fixed_weight_table():
    """测试 B3 规范: 统一权重表默认值严格为 (0.45, 0.35, 0.20)"""
    assert TrendGateMachine.DEFAULT_W_PRICE == 0.45
    assert TrendGateMachine.DEFAULT_W_NALE == 0.35
    assert TrendGateMachine.DEFAULT_W_TS == 0.20

    machine = TrendGateMachine()
    assert machine.price_weight == pytest.approx(0.45, rel=1e-6)
    assert machine.nale_weight == pytest.approx(0.35, rel=1e-6)
    assert machine.tsmom_weight == pytest.approx(0.20, rel=1e-6)


def test_trend_gate_tsmom_neutral_strict_equivalence():
    """测试 B3 核心验收标准: tsmom=None 与 tsmom=0.0 的 composite_score 逐点完全相等"""
    np.random.seed(42)
    prices = pd.Series(np.cumsum(np.random.normal(0, 5, 100)) + 1500.0)
    nale = pd.Series(np.random.uniform(-0.5, 0.8, 100))

    machine = TrendGateMachine()
    res_none = machine.evaluate_regime(prices, nale_signals=nale, tsmom_signals=None)
    res_zero = machine.evaluate_regime(prices, nale_signals=nale, tsmom_signals=pd.Series(0.0, index=prices.index))

    # 严格检验逐点差值为 0 (bitwise/exact numeric equivalence)
    score_diff = np.abs(res_none["composite_score"].values - res_zero["composite_score"].values)
    assert np.max(score_diff) == 0.0, "tsmom=None 与 tsmom=0.0 产生的 composite_score 不恒等！"
    pd.testing.assert_series_equal(res_none["trend_gate"], res_zero["trend_gate"])
    pd.testing.assert_series_equal(res_none["dynamic_hedge_ratio"], res_zero["dynamic_hedge_ratio"])


def test_trend_gate_neutral_fill_denominator_invariant():
    """测试缺失信号中性填补且严格不改变有效分母"""
    prices = pd.Series([100.0] * 50 + [100.0 - i * 1.5 for i in range(1, 60)])
    machine = TrendGateMachine()

    # 1. 全部信号缺失 -> 纯价格贡献 0.45 * 1.0 = 0.45
    res_all_none = machine.evaluate_regime(prices, nale_signals=None, tsmom_signals=None)
    assert res_all_none["composite_score"].iloc[-1] == pytest.approx(0.45, rel=1e-6)

    # 2. 传入全 0.0 信号 -> 结果必须完全一致为 0.45
    res_all_zero = machine.evaluate_regime(
        prices,
        nale_signals=pd.Series(0.0, index=prices.index),
        tsmom_signals=pd.Series(0.0, index=prices.index)
    )
    assert res_all_zero["composite_score"].iloc[-1] == pytest.approx(0.45, rel=1e-6)
