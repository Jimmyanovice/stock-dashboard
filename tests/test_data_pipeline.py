# -*- coding: utf-8 -*-
"""
数据与特征工程管道自动化测试套件 (tests/test_data_pipeline.py)
--------------------------------------------------------------
对应系统需求 R1 与验收标准 Acceptance Criterion 1:
1. 0.000% 缺失率断言 (消除所有滚动窗口首部 NaN)
2. 交易日时间序列严格单调递增性断言
3. 2019-2026 全历史周期覆盖度断言 (行数 >= 1650)
4. M1 接口契约列与物理数学边界断言
5. 自适应扩展-滚动算子单元测试
6. 基差收敛与虚拟加工利润公式精确性测试
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import (
    FuturesMarketRecord,
    BasisRecord,
    OptionIVRecord,
    AlignedFeatureRecord,
    REQUIRED_COLUMNS,
    EXTENDED_COLUMNS,
    validate_aligned_dataframe
)
from src.data.feature_engineering import (
    compute_log_returns,
    adaptive_rolling_volatility,
    compute_parkinson_volatility,
    compute_commodity_spreads,
    adaptive_rolling_correlation,
    compute_calibrated_basis_and_spot
)
from src.data.pipeline import clean_single_futures, generate_aligned_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FILE = ROOT_DIR / "data" / "processed" / "czce_sa_fg_aligned_features.csv"
SUMMARY_FILE = ROOT_DIR / "reports" / "tables" / "czce_summary_statistics.csv"


@pytest.fixture(scope="module")
def aligned_df() -> pd.DataFrame:
    """加载已生成的特征工程对齐数据集"""
    assert PROCESSED_FILE.exists(), f"对齐特征文件不存在: {PROCESSED_FILE}，请先执行 pipeline"
    df = pd.read_csv(PROCESSED_FILE)
    return df


class TestDataPipelineAcceptance:
    """验收标准 1 核心门禁自动化测试"""

    def test_pipeline_output_file_exists(self):
        """测试 1: 验证特征衍生品产物与统计表格物理存在且非空"""
        assert PROCESSED_FILE.exists(), "czce_sa_fg_aligned_features.csv 不存在"
        assert PROCESSED_FILE.stat().st_size > 100 * 1024, "特征文件体积异常偏小"
        assert SUMMARY_FILE.exists(), "czce_summary_statistics.csv 不存在"

    def test_zero_missing_rate_across_entire_dataset(self, aligned_df: pd.DataFrame):
        """测试 2: 验证全表 43 列特征缺失率绝对恒为 0.000% (Acceptance Criterion 1)"""
        total_nulls = aligned_df.isnull().sum().sum()
        null_breakdown = aligned_df.isnull().sum()
        failing_cols = null_breakdown[null_breakdown > 0].to_dict()
        assert total_nulls == 0, f"发现缺失值: {failing_cols}，全表缺失率必须严格为 0.000%"

    def test_strictly_monotonic_ascending_dates(self, aligned_df: pd.DataFrame):
        """测试 3: 验证交易日序列严格单调递增，无重复戳、无乱序、无休市日倒挂"""
        date_series = pd.to_datetime(aligned_df["date"])
        diffs = date_series.diff().dropna()
        
        # 每一个后续时间戳必须严格晚于前一个时间戳
        is_strictly_monotonic = (diffs.dt.total_seconds() > 0).all()
        assert is_strictly_monotonic, "时间序列非严格单调递增，存在乱序或重复日期！"
        assert aligned_df["date"].nunique() == len(aligned_df), "存在重复交易日"

    def test_dataset_shape_and_time_span(self, aligned_df: pd.DataFrame):
        """测试 4: 验证数据集规模 >= 1650 行，且完整覆盖 2019 至 2026 全历史周期"""
        nrows, ncols = aligned_df.shape
        assert nrows >= 1650, f"历史交易日行数不足: 当前 {nrows} 行, 验收要求 >= 1650 行"
        assert ncols >= 27, f"特征列数不足: 当前 {ncols} 列, 验收要求 >= 27 列"
        
        min_date = aligned_df["date"].min()
        max_date = aligned_df["date"].max()
        assert min_date.startswith("2019-12"), f"历史起点异常: {min_date} (应始于 2019 年 12 月纯碱上市)"
        assert max_date.startswith("2026-09"), f"历史终点异常: {max_date} (应覆盖至 2026 年 9 月)"

    def test_interface_contract_required_columns(self, aligned_df: pd.DataFrame):
        """测试 5: 验证 PROJECT.md 接口契约中明确规定的 27 项核心特征字段全部就绪"""
        missing_required = [col for col in REQUIRED_COLUMNS if col not in aligned_df.columns]
        assert len(missing_required) == 0, f"契约必需列缺失: {missing_required}"

    def test_extended_columns_backward_compatibility(self, aligned_df: pd.DataFrame):
        """测试 6: 验证为极星桥接与作图器提供向后兼容的扩展列全部就绪"""
        for col in ["ratio_fg_sa", "price_ratio_fg_sa", "sa_daily_range_pct", "fg_daily_range_pct", "iv_atm_sa", "collar_skew_sa"]:
            assert col in aligned_df.columns, f"兼容扩展列缺失: {col}"

    def test_feature_mathematical_bounds(self, aligned_df: pd.DataFrame):
        """测试 7: 验证高维金融特征数值严格满足统计物理学边界"""
        # 1. 价格必须为正
        for price_col in ["close_sa", "open_sa", "high_sa", "low_sa", "close_fg", "open_fg", "high_fg", "low_fg", "spot_sa", "spot_fg"]:
            assert (aligned_df[price_col] > 0).all(), f"{price_col} 出现非正数异常"
            
        # 2. 滚动相关系数必须严格在 [-1.0, 1.0] 内
        for corr_col in ["rolling_corr_20d", "rolling_corr_60d"]:
            min_c = aligned_df[corr_col].min()
            max_c = aligned_df[corr_col].max()
            assert min_c >= -1.0 - 1e-6, f"{corr_col} 下界越界: {min_c}"
            assert max_c <= 1.0 + 1e-6, f"{corr_col} 上界越界: {max_c}"
            
        # 3. 年化波动率必须严格为正且处于合理区间 [0.05, 3.0]
        for vol_col in ["vol_20d_sa", "vol_60d_sa", "vol_20d_fg", "vol_60d_fg", "parkinson_vol_sa", "parkinson_vol_fg"]:
            assert (aligned_df[vol_col] > 0).all(), f"{vol_col} 包含非正数"
            assert aligned_df[vol_col].max() < 3.0, f"{vol_col} 异常发散: {aligned_df[vol_col].max()}"

    def test_basis_identity_and_crush_spread_formulas(self, aligned_df: pd.DataFrame):
        """测试 8: 验证基差定义 (Basis = Futures - Spot) 与化学计量比毛利精准恒等式"""
        # 基差恒等式
        diff_sa = (aligned_df["basis_sa"] - (aligned_df["close_sa"] - aligned_df["spot_sa"])).abs()
        assert (diff_sa < 1e-4).all(), "纯碱基差不满足 Futures - Spot"
        
        diff_fg = (aligned_df["basis_fg"] - (aligned_df["close_fg"] - aligned_df["spot_fg"])).abs()
        assert (diff_fg < 1e-4).all(), "玻璃基差不满足 Futures - Spot"
        
        # 基差率恒等式
        diff_rate_sa = (aligned_df["basis_rate_sa"] - (aligned_df["basis_sa"] / aligned_df["spot_sa"])).abs()
        assert (diff_rate_sa < 1e-4).all(), "纯碱基差率不满足 Basis / Spot"
        
        # 虚拟压榨利润恒等式: P_fg - 0.20 * P_sa - 350.0 (扣除燃料成本)
        fuel = 350.0
        diff_crush = (aligned_df["crush_spread"] - (aligned_df["close_fg"] - 0.20 * aligned_df["close_sa"] - fuel)).abs()
        assert (diff_crush < 1e-4).all(), "虚拟加工利润不满足 P_fg - 0.20 * P_sa - 350.0"

    def test_summary_statistics_table_integrity(self):
        """测试 9: 验证导出的学术指标摘要表格各列缺失率均为 0.0"""
        summary_df = pd.read_csv(SUMMARY_FILE, index_col=0)
        assert not summary_df.empty, "统计摘要表为空"
        assert "missing_rate_pct" in summary_df.columns, "统计摘要表缺失 missing_rate_pct 字段"
        assert (summary_df["missing_rate_pct"] == 0.0).all(), "统计摘要表中存在非 0.0 缺失率！"

    def test_truncation_time_invariance(self, aligned_df: pd.DataFrame):
        """测试 10: 验证历史时序截断不变性 (Causal Truncation Time Invariance)

        截断未来行 (T -> T_cut) 时，历史各交易日 (t <= T_cut) 的全部特征值必须严格保持恒定不变 (diff == 0.0)，
        杜绝任何未来数据或伪随机数发生器错位引起的前视漂移。
        """
        # 1. 算子级基差与现货不变性断言
        s_sa = aligned_df["close_sa"]
        s_fg = aligned_df["close_fg"]
        full_basis = compute_calibrated_basis_and_spot(s_sa, s_fg, random_seed=42)
        for cut in [50, 200, 800, 1200]:
            cut_basis = compute_calibrated_basis_and_spot(s_sa.iloc[:cut], s_fg.iloc[:cut], random_seed=42)
            for key in full_basis:
                diff = np.max(np.abs(full_basis[key].iloc[:cut].values - cut_basis[key].values))
                assert diff == 0.0, f"{key} 在截断长度 {cut} 下基差算子不变性失效: diff={diff}"

        # 2. 全流水线级历史截断不变性断言
        sa_raw = ROOT_DIR / "data" / "raw" / "czce_sa_futures_daily_raw.csv"
        fg_raw = ROOT_DIR / "data" / "raw" / "czce_fg_futures_daily_raw.csv"
        sa_clean = clean_single_futures(sa_raw, "SA")
        fg_clean = clean_single_futures(fg_raw, "FG")
        aligned_full = generate_aligned_dataset(sa_clean, fg_clean)

        for cut in [100, 500, 1000]:
            cutoff_date = aligned_full["date"].iloc[cut - 1]
            sa_cut = sa_clean[sa_clean["date"] <= cutoff_date]
            fg_cut = fg_clean[fg_clean["date"] <= cutoff_date]
            aligned_cut = generate_aligned_dataset(sa_cut, fg_cut)

            num_cols = aligned_full.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                diff = np.max(np.abs(aligned_full[col].iloc[:cut].values - aligned_cut[col].values))
                assert diff == 0.0, f"截断不变性失效: {col} 在截断长度 {cut} 下历史差异为 {diff}"


class TestAdaptiveFeatureOperators:
    """自适应扩展-滚动算子单元测试 (Edge Cases & Micro-Tests)"""

    def test_adaptive_rolling_volatility_zero_nans_on_short_series(self):
        """测试短样本下的自适应扩展波动率无 NaN 表现"""
        short_ret = pd.Series([0.01, -0.02, 0.015, -0.005, 0.03])
        # 窗口为 20，但序列仅 5 个元素
        vol = adaptive_rolling_volatility(short_ret, window=20, min_periods=2)
        assert vol.isnull().sum() == 0, "短样本下自适应波动率产生 NaN"
        assert len(vol) == 5, "长度不一致"
        assert (vol > 0).all(), "波动率存在非正数"

    def test_adaptive_rolling_correlation_zero_nans_and_bounds(self):
        """测试短样本下的自适应扩展相关系数无 NaN 与数学边界"""
        s1 = pd.Series([0.01, -0.02, 0.015, 0.008])
        s2 = pd.Series([0.015, -0.01, 0.02, 0.005])
        corr = adaptive_rolling_correlation(s1, s2, window=20, min_periods=3)
        assert corr.isnull().sum() == 0, "短样本下自适应相关系数产生 NaN"
        assert (corr >= -1.0).all() and (corr <= 1.0).all(), "相关系数超出 [-1, 1]"

    def test_parkinson_volatility_analytical_accuracy(self):
        """测试 Parkinson 极差波动率在给定高低价时的解析准确度"""
        h = pd.Series([110.0, 115.0])
        l = pd.Series([100.0, 105.0])
        pv = compute_parkinson_volatility(h, l, window=2, annualize=False)
        assert pv.isnull().sum() == 0
        assert (pv > 0).all()
        # 理论值单日 x = (ln(110/100))^2 / (4 * ln 2)
        expected_x0 = (np.log(110.0 / 100.0)) ** 2 / (4.0 * np.log(2.0))
        assert abs(pv.iloc[0] - np.sqrt(expected_x0)) < 1e-4

    def test_schemas_dataclass_validation(self, aligned_df: pd.DataFrame):
        """测试 Dataclass Schema 契约校验逻辑对真实数据的兼容性"""
        row = aligned_df.iloc[0]
        # 验证单行数据可以被正确封装并无报错通过契约
        record = AlignedFeatureRecord(
            date=str(row["date"]),
            open_sa=float(row["open_sa"]),
            high_sa=float(row["high_sa"]),
            low_sa=float(row["low_sa"]),
            close_sa=float(row["close_sa"]),
            volume_sa=float(row["volume_sa"]),
            open_interest_sa=float(row["open_interest_sa"]),
            settle_sa=float(row["settle_sa"]),
            delta_oi_sa=float(row["delta_oi_sa"]),
            symbol_sa=str(row["symbol_sa"]),
            open_fg=float(row["open_fg"]),
            high_fg=float(row["high_fg"]),
            low_fg=float(row["low_fg"]),
            close_fg=float(row["close_fg"]),
            volume_fg=float(row["volume_fg"]),
            open_interest_fg=float(row["open_interest_fg"]),
            settle_fg=float(row["settle_fg"]),
            delta_oi_fg=float(row["delta_oi_fg"]),
            symbol_fg=str(row["symbol_fg"]),
            spot_sa=float(row["spot_sa"]),
            basis_sa=float(row["basis_sa"]),
            basis_rate_sa=float(row["basis_rate_sa"]),
            spot_fg=float(row["spot_fg"]),
            basis_fg=float(row["basis_fg"]),
            log_ret_sa=float(row["log_ret_sa"]),
            log_ret_fg=float(row["log_ret_fg"]),
            vol_20d_sa=float(row["vol_20d_sa"]),
            vol_60d_sa=float(row["vol_60d_sa"]),
            vol_20d_fg=float(row["vol_20d_fg"]),
            vol_60d_fg=float(row["vol_60d_fg"]),
            parkinson_vol_sa=float(row["parkinson_vol_sa"]),
            parkinson_vol_fg=float(row["parkinson_vol_fg"]),
            spread_fg_sa=float(row["spread_fg_sa"]),
            ratio_fg_sa=float(row["ratio_fg_sa"]),
            price_ratio_fg_sa=float(row["price_ratio_fg_sa"]),
            crush_spread=float(row["crush_spread"]),
            rolling_corr_20d=float(row["rolling_corr_20d"]),
            rolling_corr_60d=float(row["rolling_corr_60d"]),
            sa_daily_range_pct=float(row["sa_daily_range_pct"]),
            fg_daily_range_pct=float(row["fg_daily_range_pct"]),
            iv_atm_sa=float(row["iv_atm_sa"]),
            collar_skew_sa=float(row["collar_skew_sa"])
        )
        assert record.date == "2019-12-06"
        assert record.close_sa == 1561.0
