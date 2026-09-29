# -*- coding: utf-8 -*-
"""
Temporal NALE 单元与因果无偏性测试套件 (test_temporal_nale.py)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.temporal_nale import (
    TemporalConvolutionKernel,
    TemporalNALE,
    AsymmetricTemporalKernel,
    SupplyDemandAttentionSTGAT,
    TemporalNALEConfig
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FILE = ROOT_DIR / "data" / "processed" / "czce_sa_fg_aligned_features.csv"

def test_kernel_normalization():
    """测试卷积核离散权重归一化与非负性"""
    kernel = TemporalConvolutionKernel(lead_lag_days=15.0, half_life_days=20.0, kernel_window=60)
    assert np.isclose(np.sum(kernel.weights), 1.0, atol=1e-5), "卷积核权重必须严格归一化为 1"
    assert (kernel.weights >= 0).all(), "卷积核权重必须全部非负"

def test_kernel_peak_location():
    """测试高斯时滞脉冲在 tau_0 附近达到主响应极大值"""
    tau_0 = 15
    kernel = TemporalConvolutionKernel(lead_lag_days=float(tau_0), half_life_days=50.0, kernel_window=60, sigma=3.0)
    peak_idx = int(np.argmax(kernel.weights))
    assert abs(peak_idx - tau_0) <= 2, f"脉冲峰值索引 {peak_idx} 应在设计时滞 {tau_0} 附近"

def test_causal_no_lookahead():
    """核心因果性测试：更改未来时点的数据，当前及之前的所有卷积结果必须保持 100% 不变"""
    kernel = TemporalConvolutionKernel(lead_lag_days=10.0, half_life_days=20.0, kernel_window=40)
    series_original = np.random.normal(0, 1, 100)
    series_modified = series_original.copy()
    
    # 改变第 50 个点及之后的所有未来数据
    split_idx = 50
    series_modified[split_idx:] += 500.0
    
    out_orig = kernel.convolve_causal(series_original)
    out_mod = kernel.convolve_causal(series_modified)
    
    # split_idx 之前的所有卷积值必须完全一致 (零未来穿越)
    np.testing.assert_array_almost_equal(
        out_orig[:split_idx],
        out_mod[:split_idx],
        decimal=7,
        err_msg="严格因果检验失败：修改未来数据导致历史卷积值改变！"
    )

def test_temporal_nale_pipeline_execution():
    """测试端到端 Temporal NALE 产业链先行利润压力前瞻指标计算"""
    if not PROCESSED_FILE.exists():
        pytest.skip(f"跳过：未找到对齐数据文件 {PROCESSED_FILE}")
        
    df = pd.read_csv(PROCESSED_FILE)
    engine = TemporalNALE(window=60, sa_to_fg_lag=18.0, sa_half_life=25.0)
    res = engine.compute_crush_spread(df)
    
    assert len(res) == len(df), "计算结果行数必须与输入数据严格一致"
    assert "crush_spread" in res.columns
    assert "margin_pressure_index" in res.columns
    assert "nale_leading_signal" in res.columns
    
    # 检验先行信号有界性 [-1.0, 1.0]
    valid_signals = res["nale_leading_signal"].dropna()
    assert (valid_signals >= -1.0).all() and (valid_signals <= 1.0).all(), "先行信号必须严格归一化在 [-1, 1]"
    assert not np.isnan(res["crush_spread"]).any(), "裂解毛利不得含有 NaN"


def test_asymmetric_kernel_peak_transmission():
    """
    Test 1: 非对称脉冲响应核传递时滞检验
    验证成本推涨正向冲击 (Delta P >= 0) 的传导速度显著快于降价粘性负向冲击 (Delta P < 0)：
    峰值响应时间 T_peak^+ < T_peak^-，且两者符合理论参数设计（约 10 天 vs 28 天）。
    """
    kernel = AsymmetricTemporalKernel(
        tau_up=10.0,
        tau_down=28.0,
        half_life_up=14.0,
        half_life_down=35.0,
        kernel_window=60,
        sigma_up=4.0,
        sigma_down=8.0
    )
    
    # 1. 脉冲权重峰值位置检验
    assert kernel.peak_up < kernel.peak_down, (
        f"正向冲击传导峰值 ({kernel.peak_up}) 必须快于负向冲击传导峰值 ({kernel.peak_down})"
    )
    assert abs(kernel.peak_up - 10.0) <= 2.0, f"正向峰值 {kernel.peak_up} 应在 tau_up=10 附近"
    assert abs(kernel.peak_down - 28.0) <= 3.0, f"负向峰值 {kernel.peak_down} 应在 tau_down=28 附近"
    
    # 2. 动态脉冲响应卷积检验 (Impulse Response Functions)
    window = 60
    impulse_pos = np.zeros(window)
    impulse_pos[0] = 1.0
    impulse_neg = np.zeros(window)
    impulse_neg[0] = -1.0
    
    resp_pos = kernel.convolve_asymmetric(impulse_pos)
    resp_neg = kernel.convolve_asymmetric(impulse_neg)
    
    t_peak_pos = int(np.argmax(resp_pos))
    t_peak_neg = int(np.argmax(np.abs(resp_neg)))
    
    assert t_peak_pos < t_peak_neg, (
        f"正向冲击峰值时点 {t_peak_pos} 必须早于负向冲击峰值时点 {t_peak_neg}"
    )
    assert resp_pos[t_peak_pos] > 0, "正向冲击应产生正向利润/成本推涨响应"
    assert resp_neg[t_peak_neg] < 0, "负向冲击应产生负向响应"


def test_stgat_attention_weights_normalization():
    """
    Test 2: ST-GAT 图注意力权重归一化与非负性检验
    验证注意力系数严格满足 Softmax 归一化条件：sum_j alpha_{ij}(t) == 1.0 且 alpha_{ij}(t) >= 0。
    """
    stgat = SupplyDemandAttentionSTGAT(gamma=1.5, leaky_relu_alpha=0.2)
    n = 50
    np.random.seed(123)
    h_up = np.random.normal(0, 0.05, n)
    h_down = np.random.normal(0, 0.05, n)
    h_target = 0.5 * (h_up + h_down)
    div = np.random.normal(0, 1.0, n)
    
    alpha_up, alpha_down = stgat.compute_attention(
        h_upstream=h_up,
        h_downstream=h_down,
        h_target=h_target,
        divergence=div
    )
    
    assert len(alpha_up) == n and len(alpha_down) == n
    assert (alpha_up >= 0.0).all(), "上游注意力权重必须非负"
    assert (alpha_down >= 0.0).all(), "下游注意力权重必须非负"
    
    sum_weights = alpha_up + alpha_down
    np.testing.assert_allclose(
        sum_weights,
        1.0,
        rtol=1e-6,
        atol=1e-6,
        err_msg="ST-GAT 注意力权重之和必须严格归一化为 1.0"
    )


def test_stgat_supply_demand_divergence_sensitivity():
    """
    Test 3: 供需偏离度敏感性检验
    验证当下游供需偏离度 (divergence) 升高时，模型自适应增加对下游需求端的注意力权重 alpha_downstream。
    """
    stgat = SupplyDemandAttentionSTGAT(gamma=2.0, leaky_relu_alpha=0.2)
    
    # 固定节点基础特征，逐步拉大供需偏离度
    div_levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    n = len(div_levels)
    h_up = np.zeros(n)
    h_down = np.zeros(n)
    
    alpha_up, alpha_down = stgat.compute_attention(
        h_upstream=h_up,
        h_downstream=h_down,
        divergence=div_levels
    )
    
    # 验证单调递增性: div 越高，alpha_down 越大
    for i in range(len(div_levels) - 1):
        assert alpha_down[i + 1] > alpha_down[i], (
            f"供需偏离度由 {div_levels[i]} 升至 {div_levels[i+1]} 时，"
            f"下游需求注意力必须严格单调递增！({alpha_down[i]} -> {alpha_down[i+1]})"
        )
    
    # 极端偏离度检验
    assert alpha_down[-1] > 0.80, "极高下游需求偏离度下，需求注意力应占主导 (>0.80)"
    assert alpha_up[0] > 0.80, "极低下游需求偏离度（成本端主导）下，上游成本注意力应占主导 (>0.80)"


def test_causal_truncation_time_invariance():
    """
    Test 4: 严格因果截断时间不变性检验（零未来函数前视偏差）
    对长度 120 的序列截断至 70，截断前 70 天的所有计算结果（卷积响应、注意力权重、NCSI）
    必须与全量样本的前 70 天 100% 恒等；修改未来数据不得对历史造成任何扰动。
    """
    np.random.seed(42)
    n_full = 120
    split_point = 70
    
    p_sa = 1600.0 + np.cumsum(np.random.normal(0, 15, n_full))
    p_fg = 1500.0 + np.cumsum(np.random.normal(0, 10, n_full))
    inv = 80.0 + np.cumsum(np.random.normal(0, 2, n_full))
    
    df_full = pd.DataFrame({
        "close_sa": p_sa,
        "close_fg": p_fg,
        "inventory_sa": inv
    })
    
    df_trunc = df_full.iloc[:split_point].copy()
    
    df_mod_future = df_full.copy()
    df_mod_future.iloc[split_point:, df_mod_future.columns.get_loc("close_sa")] += 1000.0
    df_mod_future.iloc[split_point:, df_mod_future.columns.get_loc("close_fg")] -= 800.0
    
    engine = TemporalNALE(window=40)
    res_full = engine.compute_chain_response(df_full)
    res_trunc = engine.compute_chain_response(df_trunc)
    res_mod = engine.compute_chain_response(df_mod_future)
    
    columns_to_check = [
        "conv_upstream_cost",
        "conv_downstream_demand",
        "asym_upstream_cost",
        "asym_downstream_demand",
        "stgat_divergence_attention",
        "ncsi"
    ]
    
    for col in columns_to_check:
        # 1. 截断不变性检验
        np.testing.assert_allclose(
            res_trunc[col].values,
            res_full[col].iloc[:split_point].values,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"因果截断不变性失效：列 {col} 在截断样本与全量样本前缀不一致！"
        )
        # 2. 未来扰动不变性检验
        np.testing.assert_allclose(
            res_mod[col].iloc[:split_point].values,
            res_full[col].iloc[:split_point].values,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"前视偏差漏洞：修改未来数据导致历史列 {col} 发生漂移！"
        )


def test_backward_compatibility_and_contract_on_real_dataset():
    """
    Test 5: 真实数据集接口契约与向后兼容性检验
    在 data/processed/czce_sa_fg_aligned_features.csv 上执行 compute_chain_response
    验证所有必要列、扩展列完整生成，零 NaN 异常，且 compute_crush_spread 保持无缝兼容。
    """
    if not PROCESSED_FILE.exists():
        pytest.skip(f"跳过：未找到对齐数据文件 {PROCESSED_FILE}")
        
    df = pd.read_csv(PROCESSED_FILE)
    config = TemporalNALEConfig(
        use_asymmetric_kernel=True,
        tau_up=10.0,
        tau_down=28.0,
        half_life_up=14.0,
        half_life_down=35.0,
        use_stgat_attention=True
    )
    engine = TemporalNALE(config=config)
    res = engine.compute_chain_response(df)
    
    assert len(res) == len(df), "计算结果行数必须与真实数据严格一致"
    
    # 验证全部契约字段
    required_cols = [
        "conv_upstream_cost",
        "conv_downstream_demand",
        "asym_upstream_cost",
        "asym_downstream_demand",
        "stgat_divergence_attention",
        "ncsi",
        "crush_spread",
        "crush_margin_ratio",
        "convolved_sa_shock",
        "convolved_fg_trend",
        "margin_pressure_index",
        "nale_leading_signal"
    ]
    for col in required_cols:
        assert col in res.columns, f"契约字段缺失: {col}"
        assert not res[col].isna().any(), f"字段 {col} 存在未处理的 NaN 缺失值"
        
    # 验证 NCSI 与 Attention 物理数值边界
    assert (res["ncsi"] >= -1.0 - 1e-6).all() and (res["ncsi"] <= 1.0 + 1e-6).all(), "NCSI 必须在 [-1, 1] 有界"
    assert (res["stgat_divergence_attention"] >= 0.0).all() and (res["stgat_divergence_attention"] <= 1.0).all(), "Attention 必须在 [0, 1]"
    
    # 验证既有 compute_crush_spread 方法向后兼容执行
    res_legacy = engine.compute_crush_spread(df)
    assert len(res_legacy) == len(df)
    assert "nale_leading_signal" in res_legacy.columns

