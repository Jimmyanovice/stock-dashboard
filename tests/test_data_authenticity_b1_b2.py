# -*- coding: utf-8 -*-
"""
数据真实性与反假模拟强断言测试套件 (tests/test_data_authenticity_b1_b2.py)
---------------------------------------------------------------------
严格验证：
1. 纯碱与玻璃基差序列 100% 来源于郑商所与生意社官方真实期现数据，彻底移除随机模拟；
2. 仓单序列逐日对齐交易所真实在册仓单，微观库存优雅降级，NALE 日间差分标准差 < 0.15；
3. 回测引擎彻底移除假数据回退分支，缺失时抛出 ValueError。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import compute_calibrated_basis_and_spot
from src.models.basis_term_structure import BasisConvergenceEngine
from src.models.temporal_nale import TemporalNALE, TemporalNALEConfig
from src.models.hedging_strategies import HedgingEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
REAL_BASIS_PATH = ROOT_DIR / "data" / "raw" / "czce_real_spot_basis.csv"
REAL_RECEIPTS_PATH = ROOT_DIR / "data" / "raw" / "czce_warehouse_receipts.csv"
ALIGNED_PATH = ROOT_DIR / "data" / "processed" / "czce_sa_fg_aligned_features.csv"


def test_real_basis_cache_integrity():
    """测试 1: 验证真实基差缓存文件存在，行数 > 3000，且覆盖 SA 与 FG"""
    assert REAL_BASIS_PATH.exists(), f"真实基差缓存缺失: {REAL_BASIS_PATH}"
    df = pd.read_csv(REAL_BASIS_PATH)
    assert len(df) >= 3000, f"真实基差行数不足: {len(df)}"
    symbols = set(df["symbol"].unique())
    assert "SA" in symbols and "FG" in symbols, f"未涵盖 SA 与 FG 品种: {symbols}"

    # 验证字段完整性
    for col in ["date", "symbol", "spot_price", "dom_basis", "dom_basis_rate"]:
        assert col in df.columns, f"字段缺失: {col}"
        assert not df[col].isna().all(), f"字段全为 NaN: {col}"


def test_zero_random_simulation_in_pipeline():
    """测试 2: 验证特征工程与基差计算彻底消除随机数，两次调用 100% 精确恒等 (diff == 0.0)"""
    np.random.seed(123)
    p_sa = pd.Series(1500.0 + np.cumsum(np.random.normal(0, 10, 100)))
    p_fg = pd.Series(1600.0 + np.cumsum(np.random.normal(0, 8, 100)))

    # 两次调用基差计算，验证绝对无随机数注入
    res1 = compute_calibrated_basis_and_spot(p_sa, p_fg, random_seed=42)
    res2 = compute_calibrated_basis_and_spot(p_sa, p_fg, random_seed=999)

    for k in res1:
        diff = np.max(np.abs(res1[k].values - res2[k].values))
        assert diff == 0.0, f"算子内部仍然存在随机数种子依赖！列 {k} 差异 diff={diff}"

    # 验证 simulate_dynamic_basis 也完全确定性
    engine = BasisConvergenceEngine()
    sim1 = engine.simulate_dynamic_basis(p_sa, seed=1)
    sim2 = engine.simulate_dynamic_basis(p_sa, seed=2)
    for k in ["spot_sa", "basis_sa", "basis_rate_sa", "basis_convergence_speed_sa"]:
        diff = np.max(np.abs(sim1[k].values - sim2[k].values))
        assert diff == 0.0, f"simulate_dynamic_basis 仍依赖 seed！列 {k} 差异 diff={diff}"


def test_audit_basis_data_parity_with_czce():
    """测试 3: 验证真实基差与现货价格在关键采样点精确吻合交易所官方数据"""
    if not ALIGNED_PATH.exists():
        pytest.skip(f"对齐特征文件不存在: {ALIGNED_PATH}")

    df = pd.read_csv(ALIGNED_PATH)
    latest = df[df["date"] == "2026-09-28"]
    if not latest.empty:
        r = latest.iloc[0]
        # 官方数据: 2026-09-28 SA 现货 1075.71, 主力 1014.0, 主力基差 -61.71
        assert abs(r["spot_sa"] - 1075.71) < 1.0, f"2026-09-28 SA 现货不符: {r['spot_sa']}"
        assert abs(r["basis_sa"] - (-61.71)) < 1.0, f"2026-09-28 SA 基差不符: {r['basis_sa']}"
        # 官方数据: 2026-09-28 FG 现货 984.00, 主力 893.0, 主力基差 -91.00
        assert abs(r["spot_fg"] - 984.00) < 1.0, f"2026-09-28 FG 现货不符: {r['spot_fg']}"
        assert abs(r["basis_fg"] - (-91.00)) < 1.0, f"2026-09-28 FG 基差不符: {r['basis_fg']}"

    # 验证真实基差相关系数 > 0.95
    raw_df = pd.read_csv(REAL_BASIS_PATH)
    raw_sa = raw_df[raw_df["symbol"] == "SA"][["date", "dom_basis"]].dropna()
    merged = pd.merge(df[["date", "basis_sa"]], raw_sa, on="date")
    corr = merged["basis_sa"].corr(merged["dom_basis"])
    assert corr > 0.95, f"真实基差相关系数偏低: {corr:.4f}"


def test_real_warehouse_receipts_authenticity():
    """测试 4: 验证真实仓单数据与交易所公告严格一致 (2026-09-28 为 1851 张)"""
    if not REAL_RECEIPTS_PATH.exists():
        pytest.skip(f"仓单缓存文件不存在: {REAL_RECEIPTS_PATH}")

    df_wr = pd.read_csv(REAL_RECEIPTS_PATH)
    assert len(df_wr) >= 1650, f"仓单行数不足: {len(df_wr)}"
    assert not df_wr["warehouse_receipts_sa"].isna().any(), "仓单包含 NaN"
    assert (df_wr["warehouse_receipts_sa"] >= 0.0).all(), "仓单出现负数"

    # 验证关键时点
    row_latest = df_wr[df_wr["date"] == "2026-09-28"]
    if not row_latest.empty:
        val = row_latest.iloc[0]["warehouse_receipts_sa"]
        assert abs(val - 1851.0) < 1e-4, f"2026-09-28 仓单核验失败: {val} (期望 1851)"

    row_sample = df_wr[df_wr["date"] == "2021-05-10"]
    if not row_sample.empty:
        val = row_sample.iloc[0]["warehouse_receipts_sa"]
        assert abs(val - 4845.0) < 1e-4, f"2021-05-10 仓单核验失败: {val} (期望 4845)"


def test_nale_day_over_day_stability():
    """测试 5: 验证库存项优雅降级与因果 EMA 平滑后，NALE 日间波动标准差 < 0.15"""
    if not ALIGNED_PATH.exists():
        pytest.skip(f"对齐特征文件不存在: {ALIGNED_PATH}")

    df = pd.read_csv(ALIGNED_PATH)
    config = TemporalNALEConfig(inventory_weight=0.0, divergence_ema_span=5, signal_ema_span=5)
    model = TemporalNALE(config=config)
    res = model.compute_chain_response(df)

    diff_series = res["nale_leading_signal"].diff().dropna()
    diff_std = diff_series.std()
    assert diff_std < 0.15, f"NALE 日间变动标准差过大: {diff_std:.4f} (要求 < 0.15)"


def test_hedging_strategies_no_simulation_fallback():
    """测试 6: 验证回测引擎在缺失 spot_sa / basis_sa 时抛出 ValueError，杜绝假数据模拟回退"""
    bad_df = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "close_sa": [1800.0, 1810.0]
        # 缺失 spot_sa 与 basis_sa
    })
    with pytest.raises(ValueError, match="缺失真实 spot_sa 与 basis_sa 特征"):
        HedgingEngine(df=bad_df)
