# -*- coding: utf-8 -*-
"""
纯碱基差期限结构与便利收益模型专项测试套件 (tests/test_basis_term_structure.py)
-------------------------------------------------------------------------
涵盖 Round 1 文献驱动演化升级的 10 大核心数理与工程测试用例:
1. test_schwartz_smith_closed_form_identity: 闭式解与到期无套利终值收敛
2. test_convenience_yield_backwardation_monotonicity: 便利收益率在反向市场的单调性与持有成本平价
3. test_roll_yield_and_term_structure_slope_identity: 展期收益率与期限结构斜率相反数恒等式
4. test_dynamic_basis_convergence_speed_bounds: 动态基差收敛速度严格有界性 [0.015, 0.160]
5. test_warehouse_receipts_and_inventory_monotonicity: 仓单与库存对基差收敛速度的正向单调敏感度
6. test_tsmom_bounded_and_causal: TSMOM 动量信号有界性 [-1, 1]、零 NaN 与因果无前视性
7. test_causal_truncation_time_invariance_round1: 历史时序截断不变性 (diff == 0.0)
8. test_trend_gate_active_defense_anti_whipsaw: Trend Gate 深贴水主动防御抗假突破机制
9. test_schemas_extended_columns_round1: 数据契约强类型与扩展列完整性
10. test_end_to_end_pipeline_integration: 端到端对齐特征矩阵全量校验
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.basis_term_structure import (
    SchwartzSmithConfig,
    SchwartzSmithTwoFactorModel,
    BasisConvergenceEngine,
    TermStructureMomentum,
    compute_schwartz_smith_analytical_a
)
from src.data.schemas import (
    REQUIRED_COLUMNS,
    EXTENDED_COLUMNS,
    AlignedFeatureRecord,
    validate_aligned_dataframe
)
from src.data.feature_engineering import (
    compute_calibrated_basis_and_spot,
    compute_schwartz_smith_convenience_yield,
    compute_term_structure_slope_and_roll_yield,
    compute_term_structure_momentum
)
from src.models.trend_gate import TrendGateMachine, TrendGateStateMachine

ROOT_DIR = Path(__file__).resolve().parent.parent
ALIGNED_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "czce_sa_fg_aligned_features.csv"


def test_schwartz_smith_closed_form_identity():
    """
    Test 1: 验证 Schwartz-Smith (2000) 解析闭式解与到期收敛性。
    当 tau -> 0 时，确定性积分项 A(0) = 0，期货价格 F(t, t) 严格收敛于现货价格 S_t。
    """
    config = SchwartzSmithConfig()
    model = SchwartzSmithTwoFactorModel(config=config)
    
    # 1. 验证 A(0) == 0.0
    a_zero = compute_schwartz_smith_analytical_a(0.0, config)
    assert abs(a_zero) < 1e-12, f"A(0) 必须严格为 0: {a_zero}"
    
    # 2. 验证任意 [chi, xi] 状态下到期期货收敛于现货
    test_states = [
        (0.0, 7.5),
        (0.25, 7.8),
        (-0.15, 7.2),
        (0.40, 8.0)
    ]
    for chi, xi in test_states:
        log_spot = chi + xi
        spot = np.exp(log_spot)
        log_fut_tau0 = model.log_futures_price(chi=chi, xi=xi, tau=0.0)
        fut_tau0 = model.futures_price(chi=chi, xi=xi, tau=0.0)
        
        diff_log = abs(log_fut_tau0 - log_spot)
        diff_level = abs(fut_tau0 - spot)
        assert diff_log < 1e-6, f"对数期货价格到期收敛误差过大: {diff_log}"
        assert diff_level < 1e-4, f"期货价格到期未收敛于现货价格: fut={fut_tau0}, spot={spot}"


def test_convenience_yield_backwardation_monotonicity():
    """
    Test 2: 验证便利收益率与持有成本平价。
    在反向市场 (Backwardation, S > F, RY > 0) 中，瞬时净便利收益率 delta_t 严格大于无风险利率与仓储成本之和 (delta_t > r + u)。
    在正向市场 (Contango, S < F, RY < 0) 中，delta_t < r + u。
    """
    config = SchwartzSmithConfig(risk_free_rate=0.03, storage_cost=0.02)
    model = SchwartzSmithTwoFactorModel(config=config)
    cost_of_carry = config.risk_free_rate + config.storage_cost  # 0.05
    
    # 1. 反向市场测试 (S=2200, F=1800, tau=0.24)
    spot_back = pd.Series([2200.0] * 50)
    fut_back = pd.Series([1800.0] * 50)
    res_back = model.fit_filter_causal(spot=spot_back, futures=fut_back)
    
    assert (res_back["roll_yield"] > 0).all(), "反向市场展期收益率必须为正"
    assert (res_back["convenience_yield"] > cost_of_carry).all(), (
        f"反向市场便利收益率必须严格大于 r + u ({cost_of_carry}): 实际={res_back['convenience_yield'].mean()}"
    )
    
    # 2. 正向市场测试 (S=1800, F=2200, tau=0.24)
    spot_cont = pd.Series([1800.0] * 50)
    fut_cont = pd.Series([2200.0] * 50)
    res_cont = model.fit_filter_causal(spot=spot_cont, futures=fut_cont)
    
    assert (res_cont["roll_yield"] < 0).all(), "正向市场展期收益率必须为负"
    assert (res_cont["convenience_yield"] < cost_of_carry).all(), (
        f"正向市场便利收益率必须严格小于 r + u ({cost_of_carry}): 实际={res_cont['convenience_yield'].mean()}"
    )


def test_roll_yield_and_term_structure_slope_identity():
    """
    Test 3: 验证展期收益率与期限结构斜率严格满足相反数恒等式 beta_t^{TS} + RY_t == 0。
    """
    n = 100
    np.random.seed(123)
    spot = pd.Series(np.exp(np.cumsum(np.random.normal(0, 0.02, n)) + 7.5))
    futures = pd.Series(np.exp(np.cumsum(np.random.normal(0, 0.02, n)) + 7.4))
    
    res = compute_term_structure_slope_and_roll_yield(spot=spot, futures=futures)
    ry = res["roll_yield"]
    slope = res["term_structure_slope"]
    
    diff = (ry + slope).abs()
    assert (diff < 1e-12).all(), f"期限结构斜率与展期收益率不满足相反数恒等式: max_diff={diff.max()}"


def test_dynamic_basis_convergence_speed_bounds():
    """
    Test 4: 验证动态基差收敛速度在任何极端输入下均严格约束在 [0.015, 0.160] 物理区间内。
    """
    engine = BasisConvergenceEngine(lambda_min=0.015, lambda_max=0.160)
    
    # 构造跨越极端边界的仓单与库存测试用例
    extreme_cases = [
        # (wr, inv)
        (0.0, 10.0),            # 极低仓单 + 极低库存 (极度僵化贴水)
        (50000.0, 250.0),       # 海量仓单 + 胀库 (极速收敛)
        (6000.0, 75.0),         # 基准中位数状态
        (-1000.0, -50.0),       # 异常负数鲁棒性测试
        (100000.0, 500.0)       # 超大极端值测试
    ]
    
    for wr, inv in extreme_cases:
        wr_ser = pd.Series([wr] * 20)
        inv_ser = pd.Series([inv] * 20)
        lambda_ser = engine.compute_dynamic_convergence_speed(
            warehouse_receipts=wr_ser,
            inventory=inv_ser
        )
        assert (lambda_ser >= 0.015).all(), f"收敛速度突破下限 0.015: min={lambda_ser.min()}"
        assert (lambda_ser <= 0.160).all(), f"收敛速度突破上限 0.160: max={lambda_ser.max()}"

    # 验证基准状态下的中心收敛速度接近 0.040
    base_wr = pd.Series([6000.0] * 120)
    base_inv = pd.Series([75.0] * 120)
    base_lambda = engine.compute_dynamic_convergence_speed(
        warehouse_receipts=base_wr,
        inventory=base_inv
    )
    # 中心点附近平均值应在 0.035 ~ 0.055 区间内平滑衔接
    assert 0.035 <= base_lambda.mean() <= 0.060, f"基准收敛速度偏离经验中枢 0.040: {base_lambda.mean()}"


def test_warehouse_receipts_and_inventory_monotonicity():
    """
    Test 5: 验证仓单数量与现货库存对基差收敛速度的正向单调性 (d lambda / d WR > 0, d lambda / d Inv > 0)。
    """
    engine = BasisConvergenceEngine()
    
    # 1. 仓单单调递增敏感度测试 (固定库存)
    wr_increasing = pd.Series(np.linspace(1000.0, 20000.0, 50))
    inv_fixed = pd.Series([75.0] * 50)
    lambda_wr = engine.compute_dynamic_convergence_speed(
        warehouse_receipts=wr_increasing,
        inventory=inv_fixed
    )
    diff_wr = np.diff(lambda_wr.values)
    assert (diff_wr > 0).all(), "仓单增加未单调加速基差收敛速度 (d lambda / d WR <= 0)"

    # 2. 库存单调递增敏感度测试 (固定仓单)
    wr_fixed = pd.Series([6000.0] * 50)
    inv_increasing = pd.Series(np.linspace(30.0, 150.0, 50))
    lambda_inv = engine.compute_dynamic_convergence_speed(
        warehouse_receipts=wr_fixed,
        inventory=inv_increasing
    )
    diff_inv = np.diff(lambda_inv.values)
    assert (diff_inv > 0).all(), "库存增加未单调加速基差收敛速度 (d lambda / d Inv <= 0)"


def test_tsmom_bounded_and_causal():
    """
    Test 6: 验证 TSMOM 动量指标数值有界性 [-1.0, 1.0]、全时段零 NaN 与因果无前视性。
    """
    engine = TermStructureMomentum(smooth_window=20, zscore_window=60)
    
    # 1. 随机展期收益率序列
    np.random.seed(42)
    ry = pd.Series(np.random.normal(0.2, 0.5, 200))
    tsmom = engine.compute_tsmom_signal(ry)
    
    assert len(tsmom) == 200
    assert not tsmom.isnull().any(), "TSMOM 包含 NaN 缺失值"
    assert (tsmom >= -1.0 - 1e-6).all(), f"TSMOM 突破下限 -1.0: {tsmom.min()}"
    assert (tsmom <= 1.0 + 1e-6).all(), f"TSMOM 突破上限 1.0: {tsmom.max()}"

    # 2. 因果无前视性检验: 修改未来数据不影响历史信号
    ry_mod = ry.copy()
    split = 120
    ry_mod.iloc[split:] += 5.0  # 突发未来剧变
    tsmom_mod = engine.compute_tsmom_signal(ry_mod)
    
    diff_hist = np.max(np.abs(tsmom.iloc[:split].values - tsmom_mod.iloc[:split].values))
    assert diff_hist == 0.0, f"TSMOM 存在前视偏差: 历史差异 diff={diff_hist}"


def test_causal_truncation_time_invariance_round1():
    """
    Test 7: 验证 Round 1 新增特征列严格满足历史截断不变性 (diff == 0.0)。
    截断样本后，所有历史计算值与全样本历史值完全精确一致。
    """
    # 构造模拟序列
    np.random.seed(42)
    n = 600
    p_sa = pd.Series(np.exp(np.cumsum(np.random.normal(0, 0.015, n)) + 7.6))
    p_fg = pd.Series(np.exp(np.cumsum(np.random.normal(0, 0.015, n)) + 7.4))
    
    full_dict = compute_calibrated_basis_and_spot(p_sa, p_fg, random_seed=42)
    ss_full = compute_schwartz_smith_convenience_yield(full_dict["spot_sa"], p_sa)
    tsmom_full = compute_term_structure_momentum(ss_full["roll_yield"])
    
    for cut in [100, 300, 500]:
        cut_p_sa = p_sa.iloc[:cut]
        cut_p_fg = p_fg.iloc[:cut]
        cut_dict = compute_calibrated_basis_and_spot(cut_p_sa, cut_p_fg, random_seed=42)
        ss_cut = compute_schwartz_smith_convenience_yield(cut_dict["spot_sa"], cut_p_sa)
        tsmom_cut = compute_term_structure_momentum(ss_cut["roll_yield"])
        
        # 验证动态基差收敛特征不变性
        for k in ["basis_convergence_speed_sa", "warehouse_receipts_sa", "inventory_sa", "spot_sa", "basis_sa"]:
            diff = np.max(np.abs(full_dict[k].iloc[:cut].values - cut_dict[k].values))
            assert diff == 0.0, f"{k} 在截断长度 {cut} 下截断不变性失效: diff={diff}"
            
        # 验证 Schwartz-Smith 滤波不变性
        for col in ["chi_short", "xi_long", "convenience_yield", "roll_yield", "term_structure_slope"]:
            diff = np.max(np.abs(ss_full[col].iloc[:cut].values - ss_cut[col].values))
            assert diff == 0.0, f"{col} 在截断长度 {cut} 下卡尔曼滤波截断不变性失效: diff={diff}"
            
        # 验证 TSMOM 动量不变性
        diff_tsmom = np.max(np.abs(tsmom_full.iloc[:cut].values - tsmom_cut.values))
        assert diff_tsmom == 0.0, f"tsmom 在截断长度 {cut} 下截断不变性失效: diff={diff_tsmom}"


def test_trend_gate_active_defense_anti_whipsaw():
    """
    Test 8: 验证 Trend Gate™ 主动防御机制成功抑制深贴水环境下的假突破追空。
    
    情景构造:
    纯碱处于深度贴水 (现货坚挺，期货短暂微幅击穿 EMA 后迅速拉回)。
    - 未接入 TSMOM 主动防御时: 触发虚假破位 (gate=1, 被动追空割肉);
    - 接入 TSMOM 主动防御 (TSMOM < -0.2) 时: 成功识别基差保护并抑制信号 (gate=0, 保持低对冲)!
    """
    # 构造震荡中短暂刺穿均线的价格走势
    base_price = 2000.0
    prices = pd.Series([base_price] * 40 + [base_price - 15.0] * 5 + [base_price + 10.0] * 35)
    
    # 模拟深度贴水反向市场下的强负向 TSMOM (现货偏紧挺价)
    tsmom_deep_backwardation = pd.Series([-0.85] * len(prices))
    
    # 1. 无主动防御的基线状态机
    machine_no_defense = TrendGateMachine(
        fast_span=10,
        slow_span=25,
        hysteresis_window=2,
        nale_weight=0.0,
        active_defense=False
    )
    res_no_defense = machine_no_defense.evaluate_regime(prices)
    
    # 2. 启用深贴水主动防御的状态机
    machine_with_defense = TrendGateMachine(
        fast_span=10,
        slow_span=25,
        hysteresis_window=2,
        nale_weight=0.0,
        tsmom_weight=0.25,
        active_defense=True
    )
    res_with_defense = machine_with_defense.evaluate_regime(
        prices,
        tsmom_signals=tsmom_deep_backwardation
    )
    
    # 在刺穿均线区间 (第 42-44 日)
    # 验证主动防御使得分大幅压降
    score_no_defense = res_no_defense["composite_score"].iloc[43]
    score_with_defense = res_with_defense["composite_score"].iloc[43]
    assert score_with_defense < score_no_defense, "主动防御未成功降低下行破位压力得分"
    
    # 验证门控信号抑制效果: 未防御触发了 1.0 破位，防御后成功维持 0.0
    gate_no_def = res_no_defense["trend_gate"].iloc[43]
    gate_with_def = res_with_defense["trend_gate"].iloc[43]
    assert gate_no_def == 1.0, "未开启防御时应当触发均线破位假突破"
    assert gate_with_def == 0.0, "开启深贴水主动防御后必须成功抑制假突破追空"
    
    # 验证别名兼容性
    assert TrendGateStateMachine is TrendGateMachine


def test_schemas_extended_columns_round1():
    """
    Test 9: 验证数据契约扩展列包含全部 Round 1 新特征并满足边界校验。
    """
    round1_columns = [
        "convenience_yield_sa", "roll_yield_sa", "term_structure_slope_sa",
        "chi_short_sa", "xi_long_sa", "tsmom_sa",
        "basis_convergence_speed_sa", "warehouse_receipts_sa", "inventory_sa"
    ]
    for col in round1_columns:
        assert col in EXTENDED_COLUMNS, f"Round 1 核心字段缺失在 EXTENDED_COLUMNS: {col}"
        
    # 验证 dataclass 默认值兼容实例化
    record = AlignedFeatureRecord(
        date="2024-01-02",
        open_sa=1900.0, high_sa=1920.0, low_sa=1880.0, close_sa=1910.0,
        volume_sa=50000.0, open_interest_sa=300000.0, settle_sa=1905.0, delta_oi_sa=100.0, symbol_sa="SA",
        open_fg=1600.0, high_fg=1620.0, low_fg=1580.0, close_fg=1610.0,
        volume_fg=40000.0, open_interest_fg=250000.0, settle_fg=1605.0, delta_oi_fg=50.0, symbol_fg="FG",
        spot_sa=2400.0, basis_sa=-490.0, basis_rate_sa=-0.204,
        spot_fg=1590.0, basis_fg=20.0,
        log_ret_sa=0.005, log_ret_fg=0.004,
        vol_20d_sa=0.25, vol_60d_sa=0.28, vol_20d_fg=0.22, vol_60d_fg=0.24,
        parkinson_vol_sa=0.23, parkinson_vol_fg=0.20,
        spread_fg_sa=-300.0, ratio_fg_sa=0.843, price_ratio_fg_sa=0.843, crush_spread=1228.0,
        rolling_corr_20d=0.55, rolling_corr_60d=0.60,
        sa_daily_range_pct=2.1, fg_daily_range_pct=2.5,
        iv_atm_sa=0.275, collar_skew_sa=-0.03
    )
    assert record.basis_convergence_speed_sa == 0.04
    assert record.warehouse_receipts_sa == 6000.0
    assert record.inventory_sa == 75.0


def test_end_to_end_pipeline_integration():
    """
    Test 10: 验证真实对齐特征数据集 (czce_sa_fg_aligned_features.csv)
    包含全部 Round 1 演化列，缺失率恒为 0.000%，且全量通过 validate_aligned_dataframe。
    """
    assert ALIGNED_FEATURES_PATH.exists(), f"对齐特征文件不存在: {ALIGNED_FEATURES_PATH}"
    df = pd.read_csv(ALIGNED_FEATURES_PATH)
    
    assert len(df) >= 1650, f"数据集行数不足: {len(df)}"
    
    round1_columns = [
        "convenience_yield_sa", "roll_yield_sa", "term_structure_slope_sa",
        "chi_short_sa", "xi_long_sa", "tsmom_sa",
        "basis_convergence_speed_sa", "warehouse_receipts_sa", "inventory_sa"
    ]
    for col in round1_columns:
        assert col in df.columns, f"实际输出数据集缺失 Round 1 列: {col}"
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"{col} 存在缺失值: {null_count} (要求 0.000%)"
        
    # 执行全方位契约门禁校验
    errors = validate_aligned_dataframe(df, min_rows=1650)
    assert len(errors) == 0, f"契约门禁自检失败: {errors}"
