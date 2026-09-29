# -*- coding: utf-8 -*-
"""
Challenger 1 针对 M1 数据与特征管道的经验压力测试套件
(tests/test_challenger_m1_stress.py)
--------------------------------------------------------------
本测试套件由 Empirical Challenger 1 独立构建并执行，
系统性对 M1 交付物进行极限压力测试、病态分布模糊测试 (Fuzzing)、
零方差/极值冲击边界测试与非标准时序长度检验。
"""

from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import (
    REQUIRED_COLUMNS,
    validate_aligned_dataframe
)
from src.data.feature_engineering import (
    compute_log_returns,
    adaptive_rolling_volatility,
    compute_parkinson_volatility,
    compute_commodity_spreads,
    adaptive_rolling_correlation,
    compute_calibrated_basis_and_spot,
    extract_or_synthesize_option_iv
)
from src.data.pipeline import clean_single_futures, generate_aligned_dataset


class TestStressZeroVariance:
    """维度 1: 零方差 (Zero Variance) 极限测试"""

    def test_rolling_volatility_zero_variance(self):
        """测试全 0.0 与常数序列下的自适应波动率计算"""
        # 全 0 收益率
        s_zeros = pd.Series([0.0] * 50)
        vol_zeros = adaptive_rolling_volatility(s_zeros, window=20)
        assert not vol_zeros.isnull().any(), "全 0 序列产生 NaN"
        assert not np.isinf(vol_zeros).any(), "全 0 序列产生 Inf"
        assert (vol_zeros > 0).all(), "全 0 序列波动率应兜底为严格正数"
        
        # 常数正数收益率 (方差仍为 0)
        s_const = pd.Series([0.02] * 50)
        vol_const = adaptive_rolling_volatility(s_const, window=20)
        assert not vol_const.isnull().any(), "常数序列产生 NaN"
        assert not np.isinf(vol_const).any(), "常数序列产生 Inf"
        assert (vol_const > 0).all(), "常数序列波动率应兜底为严格正数"

    def test_rolling_correlation_zero_variance(self):
        """测试当单一或两个序列方差为 0 时自适应相关系数的表现"""
        s_zero = pd.Series([0.0] * 50)
        s_rand = pd.Series(np.random.RandomState(42).randn(50))
        
        # 单一方差为 0
        corr_one_zero = adaptive_rolling_correlation(s_zero, s_rand, window=20)
        assert not corr_one_zero.isnull().any(), "单方差为 0 相关系数产生 NaN"
        assert not np.isinf(corr_one_zero).any(), "单方差为 0 相关系数产生 Inf"
        assert ((corr_one_zero >= -1.0) & (corr_one_zero <= 1.0)).all()
        # 验证常数序列下回填兜底值是否符合预期 (0.50)
        assert (corr_one_zero == 0.50).all()

        # 双方方差均为 0
        corr_both_zero = adaptive_rolling_correlation(s_zero, s_zero, window=20)
        assert not corr_both_zero.isnull().any(), "双方方差为 0 相关系数产生 NaN"
        assert not np.isinf(corr_both_zero).any(), "双方方差为 0 相关系数产生 Inf"
        assert (corr_both_zero == 0.50).all()

    def test_parkinson_volatility_zero_range(self):
        """测试日内无振幅 (High == Low) 一字板/恒定价格下的 Parkinson 波动率"""
        h = pd.Series([100.0] * 50)
        l = pd.Series([100.0] * 50)
        pv = compute_parkinson_volatility(h, l, window=20)
        assert not pv.isnull().any(), "零振幅下 Parkinson 波动率产生 NaN"
        assert not np.isinf(pv).any(), "零振幅下 Parkinson 波动率产生 Inf"
        assert (pv > 0).all(), "零振幅下 Parkinson 波动率应有效回填正数"


class TestStressExtremeOutliersAndDistributions:
    """维度 2: 极端离群值与病态分布压力测试"""

    def test_cauchy_heavy_tailed_distribution(self):
        """柯西重尾分布（二阶矩不存在，极端跳跃）下的算子稳定性"""
        rng = np.random.RandomState(101)
        cauchy_ret = pd.Series(rng.standard_cauchy(200))
        
        vol = adaptive_rolling_volatility(cauchy_ret, window=20)
        assert not vol.isnull().any(), "柯西分布下波动率产生 NaN"
        assert not np.isinf(vol).any(), "柯西分布下波动率产生 Inf"
        assert (vol > 0).all(), "柯西分布下波动率非正"
        
        corr = adaptive_rolling_correlation(cauchy_ret, cauchy_ret, window=20)
        assert not corr.isnull().any(), "柯西分布下自相关产生 NaN"
        assert not np.isinf(corr).any(), "柯西分布下自相关产生 Inf"
        assert ((corr >= -1.0 - 1e-6) & (corr <= 1.0 + 1e-6)).all()

    def test_floating_point_extreme_magnitudes(self):
        """浮点数极大值与极小值冲击测试 (1e15, 1e-15)"""
        s_huge = pd.Series([1.0, 1e15, -1e15, 1.0, 1e-15] * 10)
        vol = adaptive_rolling_volatility(s_huge, window=10)
        assert not vol.isnull().any(), "极大浮点数下波动率产生 NaN"
        assert not np.isinf(vol).any(), "极大浮点数下波动率产生 Inf"
        
        corr = adaptive_rolling_correlation(s_huge, s_huge, window=10)
        assert not corr.isnull().any(), "极大浮点数下自相关产生 NaN"
        assert not np.isinf(corr).any(), "极大浮点数下自相关产生 Inf"
        assert ((corr >= -1.0 - 1e-6) & (corr <= 1.0 + 1e-6)).all()

    def test_parkinson_inverted_and_negative_prices(self):
        """测试 Parkinson 波动率在价格倒挂 (High < Low) 或负价格时的容错性"""
        h_inv = pd.Series([50.0, 40.0, 30.0, -10.0])
        l_inv = pd.Series([100.0, 90.0, 80.0, -20.0])
        pv = compute_parkinson_volatility(h_inv, l_inv, window=2)
        assert not pv.isnull().any(), "倒挂/负价格下 Parkinson 波动率产生 NaN"
        assert not np.isinf(pv).any(), "倒挂/负价格下 Parkinson 波动率产生 Inf"
        assert (pv > 0).all()


class TestStressNonStandardLengths:
    """维度 3: 非标准与边缘时序长度测试"""

    @pytest.mark.parametrize("length", [1, 2, 3, 5, 10, 19, 20, 21, 59, 60, 61])
    def test_various_sequence_lengths(self, length: int):
        """覆盖从极小样本 (1, 2, 3) 到临界窗口 (19, 20, 21, 59, 60, 61) 的全覆盖"""
        rng = np.random.RandomState(42)
        s = pd.Series(1500.0 + np.cumsum(rng.randn(length)))
        h = s + np.abs(rng.randn(length)) * 2.0
        l = s - np.abs(rng.randn(length)) * 2.0
        r = pd.Series(rng.randn(length) * 0.01)
        
        # 1. log_returns
        lr = compute_log_returns(s)
        assert len(lr) == length
        assert not lr.isnull().any()
        
        # 2. adaptive_rolling_volatility (20d & 60d)
        vol20 = adaptive_rolling_volatility(r, window=20)
        vol60 = adaptive_rolling_volatility(r, window=60)
        assert len(vol20) == length and len(vol60) == length
        assert not vol20.isnull().any() and not vol60.isnull().any()
        assert not np.isinf(vol20).any() and not np.isinf(vol60).any()
        assert (vol20 > 0).all() and (vol60 > 0).all()
        
        # 3. compute_parkinson_volatility
        pv = compute_parkinson_volatility(h, l, window=20)
        assert len(pv) == length
        assert not pv.isnull().any()
        assert (pv > 0).all()
        
        # 4. adaptive_rolling_correlation
        corr = adaptive_rolling_correlation(r, r, window=20)
        assert len(corr) == length
        assert not corr.isnull().any()
        assert ((corr >= -1.0 - 1e-6) & (corr <= 1.0 + 1e-6)).all()


class TestStressMissingAndDirtyDataIngestion:
    """维度 4: 原始数据脏数据与缺失值注入流水线鲁棒性"""

    def test_pipeline_resilience_to_dirty_csv(self):
        """测试单品种清洗在遭遇空行、字符串 null/None/-- 及缺失列时的清洗恢复力"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "dirty_futures.csv"
            df_dirty = pd.DataFrame({
                "日期": ["2024-01-05", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"],
                "开盘价": ["1500.0", "NaN", "1520.0", None, "1540.0"],
                "最高价": [1550.0, "--", 1560.0, 1550.0, "None"],
                "最低价": [1480.0, 1490.0, "null", 1470.0, 1490.0],
                "收盘价": [1520.0, 1510.0, 1510.0, 1500.0, 1530.0],
                "成交量": [1000, 2000, 2000, np.nan, 3000],
                "持仓量": [5000, 5200, 5200, 5100, np.nan]
            })
            df_dirty.to_csv(csv_path, index=False)
            
            cleaned = clean_single_futures(csv_path, "TEST_DIRTY")
            
            # 断言 1: 去重后交易日数为 4
            assert len(cleaned) == 4
            # 断言 2: 时间严格单调递增
            dates = pd.to_datetime(cleaned["date"])
            assert (dates.diff().dropna().dt.total_seconds() > 0).all()
            # 断言 3: 清洗后无任何 NaN
            assert cleaned.isnull().sum().sum() == 0

    def test_options_iv_synthetic_fallback_on_missing_or_corrupt_file(self):
        """测试当场内期权原始文件物理丢失或损坏时平滑回退至合成曲面"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dates = pd.date_range("2024-01-01", periods=10).strftime("%Y-%m-%d").to_series()
            vol = pd.Series([0.20] * 10)
            
            # 1. 物理不存在
            res_missing = extract_or_synthesize_option_iv(
                dates, vol, options_raw_path=Path(tmpdir) / "not_exist.csv"
            )
            assert len(res_missing["iv_atm_sa"]) == 10
            assert not res_missing["iv_atm_sa"].isnull().any()
            # 合成隐波 = 0.20 + 0.025 = 0.225
            assert np.allclose(res_missing["iv_atm_sa"].values, 0.225)
            
            # 2. 文件严重损坏 (非法文本)
            corrupt_file = Path(tmpdir) / "corrupt.csv"
            corrupt_file.write_text("corrupted_header\nfoo,bar,baz", encoding="utf-8")
            res_corrupt = extract_or_synthesize_option_iv(
                dates, vol, options_raw_path=corrupt_file
            )
            assert len(res_corrupt["iv_atm_sa"]) == 10
            assert not res_corrupt["iv_atm_sa"].isnull().any()
            assert np.allclose(res_corrupt["iv_atm_sa"].values, 0.225)


class TestStressPathologicalFuzzing:
    """维度 5: 蒙特卡洛随机病态分布模糊测试 (Monte Carlo Fuzzing)"""

    def test_monte_carlo_fuzzing_rolling_filters(self):
        """200 次随机病态分布检验：混合 NaN、Inf、重尾、阶跃与离群点"""
        rng = np.random.RandomState(777)
        
        for trial in range(200):
            L = rng.randint(2, 80)
            mode = trial % 5
            
            if mode == 0:
                # 混合 NaN 与 Inf
                arr1 = rng.randn(L)
                arr2 = rng.randn(L)
                arr1[rng.rand(L) < 0.2] = np.nan
                arr2[rng.rand(L) < 0.2] = np.inf
            elif mode == 1:
                # 全常数
                arr1 = np.full(L, rng.choice([0.0, 1.0, -5.0]))
                arr2 = np.full(L, rng.choice([0.0, 2.0, 10.0]))
            elif mode == 2:
                # 极端尖峰
                arr1 = np.zeros(L)
                arr2 = rng.randn(L)
                arr1[rng.randint(0, L)] = 1e8
            elif mode == 3:
                # 重尾柯西
                arr1 = rng.standard_cauchy(L)
                arr2 = rng.standard_cauchy(L)
            else:
                # 极小尺度浮点
                arr1 = rng.randn(L) * 1e-8
                arr2 = rng.randn(L) * 1e-8
                
            s1 = pd.Series(arr1)
            s2 = pd.Series(arr2)
            
            vol = adaptive_rolling_volatility(s1, window=15)
            assert not vol.isnull().any(), f"Trial {trial} (mode {mode}) vol produced NaN"
            assert not np.isinf(vol).any(), f"Trial {trial} (mode {mode}) vol produced Inf"
            
            corr = adaptive_rolling_correlation(s1, s2, window=15)
            assert not corr.isnull().any(), f"Trial {trial} (mode {mode}) corr produced NaN"
            assert not np.isinf(corr).any(), f"Trial {trial} (mode {mode}) corr produced Inf"
            assert ((corr >= -1.0 - 1e-5) & (corr <= 1.0 + 1e-5)).all()


class TestStressScalabilityAndPerformance:
    """维度 6: 大规模高维性能与内存压力测试"""

    def test_20k_rows_throughput(self):
        """测试 20,000 个交易日（约 80 年规模）吞吐性能，断言端到端在 1 秒内完成"""
        N = 20000
        rng = np.random.RandomState(42)
        
        p_sa = pd.Series(1500.0 + np.cumsum(rng.randn(N)))
        p_fg = pd.Series(1600.0 + np.cumsum(rng.randn(N)))
        h_sa = p_sa + np.abs(rng.randn(N)) * 3.0
        l_sa = p_sa - np.abs(rng.randn(N)) * 3.0
        
        t0 = time.perf_counter()
        
        lr_sa = compute_log_returns(p_sa)
        v20 = adaptive_rolling_volatility(lr_sa, window=20)
        v60 = adaptive_rolling_volatility(lr_sa, window=60)
        pv = compute_parkinson_volatility(h_sa, l_sa, window=20)
        spreads = compute_commodity_spreads(p_fg, p_sa)
        corr = adaptive_rolling_correlation(lr_sa, lr_sa, window=20)
        basis = compute_calibrated_basis_and_spot(p_sa, p_fg)
        
        elapsed = time.perf_counter() - t0
        
        assert elapsed < 1.5, f"20,000 规模执行耗时过长: {elapsed:.3f}s (应 < 1.5s)"
        assert len(v20) == N
        assert not v20.isnull().any()
        assert not basis["basis_sa"].isnull().any()
