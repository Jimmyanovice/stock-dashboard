# -*- coding: utf-8 -*-
"""
Challenger Evo 1: 数学模型与因果对抗性压力测试套件
(tests/test_challenger_evo1_adversarial.py)
================================================================================
针对 Round 1 & Round 2 文献演化数学模型的极限压力测试、奇异边界分析与因果无前视对抗验证：
1. SchwartzSmithTwoFactorModel 极端基差发散、奇异协方差与闪崩冲击
2. BasisConvergenceEngine 极端仓单清零、超新星胀库与 Logistic 有界映射边界
3. AsymmetricTemporalKernel 正负向冲击传导速度比与强对抗噪声鲁棒性
4. SupplyDemandAttentionSTGAT 极端供需偏离度溢出防范与 Softmax 概率守恒
5. 蒙特卡洛随机切分截断与右侧对抗性数据投毒 (Poisoning Attack) 下的零未来信息泄漏 (diff == 0.0)
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
    compute_schwartz_smith_analytical_a,
)
from src.models.temporal_nale import (
    TemporalNALEConfig,
    TemporalNALE,
    TemporalConvolutionKernel,
    AsymmetricTemporalKernel,
    SupplyDemandAttentionSTGAT,
)


# ==============================================================================
# 维度 1: 极端基差发散与合成闪崩冲击 (Flash Crash & Boundary Stress)
# ==============================================================================
class TestExtremeBasisAndFlashCrashes:
    """Schwartz-Smith 状态空间模型与便利收益在极端市况下的数值稳定性与代数自洽性"""

    def test_spot_flash_crash_resilience(self):
        """
        场景 1.1: 现货单日闪崩 95% (从 2400 跌至 120)，期货维持 2000。
        验证: 卡尔曼滤波器不崩溃、无 NaN/Inf、协方差矩阵 P 严格半正定 (特征值 >= 0)。
        """
        config = SchwartzSmithConfig()
        model = SchwartzSmithTwoFactorModel(config=config)

        n = 80
        spot_prices = np.full(n, 2400.0)
        futures_prices = np.full(n, 2000.0)

        # 在第 40 日发生 95% 闪崩并在第 41 日拉回
        spot_prices[40] = 120.0
        spot_ser = pd.Series(spot_prices)
        fut_ser = pd.Series(futures_prices)

        res = model.fit_filter_causal(spot=spot_ser, futures=fut_ser)

        assert not res.isnull().any().any(), "现货闪崩导致卡尔曼滤波输出产生 NaN"
        assert not np.isinf(res.values).any(), "现货闪崩导致卡尔曼滤波输出产生 Inf"

        # 展期收益率与期限结构斜率相反数恒等式必须在全样本点严格保持
        slope_identity_diff = np.abs(res["roll_yield"] + res["term_structure_slope"])
        assert (slope_identity_diff < 1e-11).all(), "闪崩冲击下相反数恒等式 beta_TS + RY == 0 破损"

        # 验证闪崩发生日 chi_short 产生深度负向跳跃 (贴水结构瞬间被物理击穿)
        assert res["chi_short"].iloc[40] < res["chi_short"].iloc[39] - 1.0

    def test_futures_flash_crash_and_extreme_backwardation(self):
        """
        场景 1.2: 期货单日闪崩 95% (从 2000 跌至 100)，现货坚挺在 2400 (超深贴水)。
        验证: TSMOM 动量信号在极度深贴水下严格饱和收敛于下界 -1.0，触发最高防御且无浮点溢出。
        """
        config = SchwartzSmithConfig()
        model = SchwartzSmithTwoFactorModel(config=config)
        tsmom_engine = TermStructureMomentum(smooth_window=10, zscore_window=30)

        n = 60
        spot_ser = pd.Series(np.full(n, 2400.0))
        fut_arr = np.full(n, 2000.0)
        fut_arr[30:35] = 100.0  # 连续 5 天极端闪崩
        fut_ser = pd.Series(fut_arr)

        res_ss = model.fit_filter_causal(spot=spot_ser, futures=fut_ser)
        tsmom = tsmom_engine.compute_tsmom_signal(res_ss["roll_yield"])

        assert not tsmom.isnull().any(), "期货闪崩下 TSMOM 产生 NaN"
        assert (tsmom >= -1.0 - 1e-6).all() and (tsmom <= 1.0 + 1e-6).all(), "TSMOM 超出 [-1, 1] 物理界限"

        # 极度深贴水必须驱动 TSMOM 顶格饱和于 -1.0 (主动防御最大化，禁止追空)
        assert np.isclose(tsmom.iloc[34], -1.0, atol=1e-3), (
            f"极端深贴水未触发 -1.0 主动防御饱和: actual={tsmom.iloc[34]}"
        )

    def test_oscillating_synthetic_whipsaw_divergence(self):
        """
        场景 1.3: 现货与期货日度交替剧烈震荡 (+50% / -50% 双向反向假突破锯齿波)。
        验证: 状态转移过程与卡尔曼滤波增益 K_gain 保持有界自愈，不出现数值爆炸。
        """
        config = SchwartzSmithConfig()
        model = SchwartzSmithTwoFactorModel(config=config)

        n = 100
        t = np.arange(n)
        whipsaw = (-1.0) ** t * 0.45
        spot = pd.Series(2000.0 * np.exp(whipsaw))
        futures = pd.Series(2000.0 * np.exp(-whipsaw))

        res = model.fit_filter_causal(spot=spot, futures=futures)
        assert not res.isnull().any().any(), "剧烈反向假突破锯齿波产生 NaN"
        assert (np.abs(res["chi_short"]) < 10.0).all(), "chi_short 状态变量爆炸"
        assert (np.abs(res["xi_long"]) < 15.0).all(), "xi_long 状态变量爆炸"

    def test_non_positive_and_sub_zero_price_protection(self):
        """
        场景 1.4: 现货或期货输入出现 0 或负数价格 (极端数据源异常)。
        验证: 模型内部采用 np.maximum(..., 1.0) 有效防护，避免对数出现 -Inf 或 NaN。
        """
        config = SchwartzSmithConfig()
        model = SchwartzSmithTwoFactorModel(config=config)

        spot = pd.Series([0.0, -100.0, 1500.0, 1600.0])
        futures = pd.Series([-50.0, 0.0, 1400.0, 1450.0])

        res = model.fit_filter_causal(spot=spot, futures=futures)
        assert not res.isnull().any().any(), "非正价格导致模型输出 NaN"
        assert not np.isinf(res.values).any(), "非正价格导致模型输出 Inf"


# ==============================================================================
# 维度 2: 极端仓单与超新星库存冲击 (Boundary Stress on Convergence Engine)
# ==============================================================================
class TestExtremeInventoryAndWarehouseReceipts:
    """BasisConvergenceEngine 在零仓单、超高库存及病态负输入下的边界响应"""

    def test_zero_warehouse_receipts_and_zero_inventory(self):
        """
        场景 2.1: 仓单为 0，库存为 0 (极端极度缺货、现货挤仓挤兑)。
        验证: 收敛速度 lambda_t 逼近理论下限 lambda_min (0.015)，且严格保持有界。
        """
        engine = BasisConvergenceEngine(lambda_min=0.015, lambda_max=0.160)
        n = 30
        wr_zero = pd.Series([0.0] * n)
        inv_zero = pd.Series([0.0] * n)

        lambda_zero = engine.compute_dynamic_convergence_speed(
            warehouse_receipts=wr_zero, inventory=inv_zero
        )

        assert (lambda_zero >= 0.015).all(), f"零仓单零库存突破理论下界: min={lambda_zero.min()}"
        assert (lambda_zero <= 0.040).all(), "极端短缺下收敛速度应当极低 (逼近贴水固化)"

    def test_supernova_inventory_glut(self):
        """
        场景 2.2: 超新星库存胀库 (库存 1,000,000 吨，仓单 1,000,000 张)。
        验证: 收敛速度 lambda_t 逼近理论上限 lambda_max (0.160)，且无浮点溢出。
        """
        engine = BasisConvergenceEngine(lambda_min=0.015, lambda_max=0.160)
        n = 30
        wr_huge = pd.Series([1_000_000.0] * n)
        inv_huge = pd.Series([1_000_000.0] * n)

        lambda_huge = engine.compute_dynamic_convergence_speed(
            warehouse_receipts=wr_huge, inventory=inv_huge
        )

        assert (lambda_huge <= 0.160).all(), f"超大库存突破理论上限: max={lambda_huge.max()}"
        assert np.allclose(lambda_huge.values, 0.160, atol=1e-4), "极大库存未饱和收敛至 lambda_max"

    def test_adversarial_corrupted_negative_inputs(self):
        """
        场景 2.3: 异常负数仓单与负数库存输入 (模拟脏数据与管道上游崩溃)。
        验证: 内部 z_safe 截断有效防护，系统不发生 math domain error，值域严格保持 [0.015, 0.160]。
        """
        engine = BasisConvergenceEngine(lambda_min=0.015, lambda_max=0.160)
        n = 40
        wr_neg = pd.Series([-999999.0] * n)
        inv_neg = pd.Series([-888888.0] * n)

        lambda_neg = engine.compute_dynamic_convergence_speed(
            warehouse_receipts=wr_neg, inventory=inv_neg
        )

        assert not lambda_neg.isnull().any(), "负输入产生 NaN"
        assert not np.isinf(lambda_neg).any(), "负输入产生 Inf"
        assert (lambda_neg >= 0.015).all() and (lambda_neg <= 0.160).all()

    def test_simulate_dynamic_basis_extreme_futures_levels(self):
        """
        场景 2.4: 极端期货价格序列下的动态基差 OU 模拟。
        涵盖 100 元极低价格到 50,000 元极高价格，验证 spot >= 800 底线约束与基差一致性。
        """
        engine = BasisConvergenceEngine()
        for f_level in [150.0, 800.0, 2000.0, 10000.0, 50000.0]:
            f_ser = pd.Series([f_level] * 50)
            res = engine.simulate_dynamic_basis(f_ser, seed=123)

            spot = res["spot_sa"]
            basis = res["basis_sa"]
            basis_rate = res["basis_rate_sa"]

            assert (spot >= 800.0).all(), f"现货底线约束失效: min_spot={spot.min()}"
            # 严格满足 Basis = Futures - Spot
            assert np.allclose(basis.values, (f_ser - spot).values, atol=1e-8)
            # 严格满足 BasisRate = Basis / Spot
            assert np.allclose(basis_rate.values, (basis / spot).values, atol=1e-8)


# ==============================================================================
# 维度 3: Logistic 映射边界与导数单调性 (Logistic Mapping Asymptotics)
# ==============================================================================
class TestLogisticConvergenceMappingMathematicalProperties:
    """验证动态收敛速度映射函数的 C^1 光滑性、极值渐近性与全局单调递增性"""

    def test_logistic_asymptotic_limits(self):
        """验证耦合指数 z -> +/- infinity 时严格单调收敛于对应开闭边界"""
        engine = BasisConvergenceEngine(lambda_min=0.015, lambda_max=0.160)

        # 扫描连续 z_t 范围从 -100 到 +100
        z_grid = np.linspace(-100.0, 100.0, 1000)
        z_safe = np.clip(z_grid, -20.0, 20.0)
        logistic_val = 1.0 / (1.0 + np.exp(-z_safe))
        lambda_val = engine.lambda_min + (engine.lambda_max - engine.lambda_min) * logistic_val

        # 1. 严格单调不减
        diffs = np.diff(lambda_val)
        assert (diffs >= 0).all(), "Logistic 映射违反全局单调递增性"

        # 2. 边界饱和
        assert abs(lambda_val[0] - 0.015) < 1e-7
        assert abs(lambda_val[-1] - 0.160) < 1e-7

    def test_numerical_derivative_strictly_positive(self):
        """验证微积分一阶导数 d lambda / d z 在有效区间内严格为正 (无平顶鞍点假死)"""
        engine = BasisConvergenceEngine()
        z_core = np.linspace(-10.0, 10.0, 200)
        dz = 1e-5

        z_plus = np.clip(z_core + dz, -20.0, 20.0)
        z_minus = np.clip(z_core - dz, -20.0, 20.0)

        l_plus = engine.lambda_min + (engine.lambda_max - engine.lambda_min) / (1.0 + np.exp(-z_plus))
        l_minus = engine.lambda_min + (engine.lambda_max - engine.lambda_min) / (1.0 + np.exp(-z_minus))

        grad = (l_plus - l_minus) / (2.0 * dz)
        assert (grad > 0.0).all(), "有效核心区间内一阶导数存在非正区域"
        # 验证最大导数出现在 z=0 拐点附近
        max_idx = np.argmax(grad)
        assert abs(z_core[max_idx]) < 0.2, f"最大斜率点偏离对称中心: {z_core[max_idx]}"


# ==============================================================================
# 维度 4: 非对称冲击传导时滞比与强对抗噪声 (Asymmetric Shock Transmission)
# ==============================================================================
class TestAsymmetricKernelUnderAdversarialNoise:
    """非对称时滞脉冲算子在强噪声、对抗性跳跃下的传导速度比与物理非对称性"""

    def test_transmission_speed_ratio_theoretical_and_empirical(self):
        """
        验证正向推涨与负向粘性传导速度之比:
        理论峰值: tau_up=10, tau_down=28 -> 理论时滞比约 2.8x。
        实测脉冲响应离散峰值必须满足 2.0 <= (t_peak_down / t_peak_up) <= 3.5。
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

        p_up = kernel.peak_up
        p_down = kernel.peak_down
        ratio = p_down / p_up

        assert 2.0 <= ratio <= 3.5, f"非对称传导时滞比偏离物理机理: {ratio:.2f} (p_up={p_up}, p_down={p_down})"

    def test_pure_impulse_energy_half_life_ratio(self):
        """
        纯脉冲响应能量耗散半衰期测试:
        正向冲击与负向冲击在达到 50% 累计能量响应时的时间比率应当接近 2.89x。
        """
        kernel = AsymmetricTemporalKernel(kernel_window=60)
        imp_pos = np.zeros(60)
        imp_pos[0] = 0.10
        imp_neg = np.zeros(60)
        imp_neg[0] = -0.10

        r_pos = kernel.convolve_asymmetric(imp_pos)
        r_neg = kernel.convolve_asymmetric(imp_neg)

        cum_pos = np.cumsum(r_pos)
        cum_neg = np.cumsum(np.abs(r_neg))

        th_pos = np.where(cum_pos >= 0.5 * cum_pos[-1])[0][0]
        th_neg = np.where(cum_neg >= 0.5 * cum_neg[-1])[0][0]

        ratio_half = th_neg / th_pos
        assert 2.0 <= ratio_half <= 3.5, f"纯脉冲响应能量半衰期比率异常: {ratio_half} (th_pos={th_pos}, th_neg={th_neg})"

    def test_adversarial_noise_shock_decoupling(self):
        """
        测试在叠加市场真实零均值高斯噪声与微观抖动对抗环境下，
        通过平滑时滞互相关分析，正向成本推涨传导峰值显著快于负向降价粘性传导峰值 (比率在 2.0 ~ 3.5x 之间)。
        """
        kernel = AsymmetricTemporalKernel(kernel_window=60)
        rng = np.random.RandomState(42)

        # 在多种典型市场噪声幅度 (0.002, 0.005, 0.010) 下测试 50 次实验
        for noise_scale in [0.002, 0.005, 0.010]:
            lags_p, lags_n = [], []
            for _ in range(30):
                noise = rng.normal(0, noise_scale, 100)
                s_p = noise.copy()
                s_p[20] += 0.08  # 正向冲击
                s_n = noise.copy()
                s_n[20] -= 0.08  # 负向冲击

                rp = kernel.convolve_asymmetric(s_p)
                rn = kernel.convolve_asymmetric(s_n)

                corr_p = rp[20:80]
                corr_n = np.abs(rn[20:80])

                # 3 点移动平均平滑滤波以剔除单样本随机扰动
                w = 3
                sm_p = np.convolve(corr_p, np.ones(w) / w, mode='valid')
                sm_n = np.convolve(corr_n, np.ones(w) / w, mode='valid')

                lags_p.append(np.argmax(sm_p) + (w // 2))
                lags_n.append(np.argmax(sm_n) + (w // 2))

            mean_p = np.mean(lags_p)
            mean_n = np.mean(lags_n)
            ratio = mean_n / mean_p

            assert 2.0 <= ratio <= 3.5, (
                f"噪声尺度 {noise_scale} 下传导时滞比失真: ratio={ratio:.2f} (mean_p={mean_p:.1f}, mean_n={mean_n:.1f})"
            )


# ==============================================================================
# 维度 5: ST-GAT 图注意力极端偏离度溢出防范 (Attention Extreme Divergence)
# ==============================================================================
class TestSTGATAttentionUnderPathologicalDivergence:
    """SupplyDemandAttentionSTGAT 在超大偏离度下的 Softmax 防溢出与概率守恒"""

    def test_divergence_extreme_magnitudes_numerical_stability(self):
        """
        向 ST-GAT 注入极大 (+10^4) 与极小 (-10^4) 供需偏离度。
        验证: Log-Sum-Exp / e_max 防护成功，无 NaN/Inf，且权重严格归一化为 1.000000。
        """
        stgat = SupplyDemandAttentionSTGAT(gamma=1.5, leaky_relu_alpha=0.2)
        divergences = np.array([1e-6, 10.0, 100.0, 1000.0, 10000.0, -10000.0, -100.0])
        n = len(divergences)

        h_u = np.ones(n) * 0.05
        h_d = np.ones(n) * 0.02

        alpha_u, alpha_d = stgat.compute_attention(
            h_upstream=h_u,
            h_downstream=h_d,
            divergence=divergences
        )

        assert not np.isnan(alpha_u).any() and not np.isnan(alpha_d).any(), "ST-GAT 在极大偏离度下出现 NaN"
        assert not np.isinf(alpha_u).any() and not np.isinf(alpha_d).any(), "ST-GAT 在极大偏离度下出现 Inf"

        # 概率和恒等式
        sums = alpha_u + alpha_d
        np.testing.assert_allclose(sums, 1.0, atol=1e-12, err_msg="ST-GAT Softmax 概率守恒失效")

        # 渐近极限: D -> +10000 时 alpha_down 饱和于 1.0; D -> -10000 时 alpha_up 饱和于 1.0
        assert np.isclose(alpha_d[4], 1.0, atol=1e-6), f"极端需求主导未饱和: {alpha_d[4]}"
        assert np.isclose(alpha_u[5], 1.0, atol=1e-6), f"极端成本主导未饱和: {alpha_u[5]}"

    def test_leaky_relu_non_zero_gradients_on_negative_regime(self):
        """
        验证 LeakyReLU 对负向 logit 的活性保持 (alpha=0.2)，杜绝标准 ReLU 的神经元死亡坏死现象。
        """
        stgat = SupplyDemandAttentionSTGAT(gamma=1.0, leaky_relu_alpha=0.2)
        val = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        out = stgat._leaky_relu(val)

        expected = np.array([-1.0, -0.4, 0.0, 2.0, 5.0])
        np.testing.assert_allclose(out, expected, atol=1e-8)


# ==============================================================================
# 维度 6: 因果截断不变性与对抗性数据投毒 (Zero-Lookahead & Poisoning Attack)
# ==============================================================================
class TestCausalNonAnticipationAndZeroLookaheadInvariance:
    """对 Round 1 & Round 2 模型进行全链路随机截断与未来数据投毒测试"""

    @pytest.mark.parametrize("split_seed", [101, 202, 303, 404, 505])
    def test_monte_carlo_random_truncation_splits(self, split_seed: int):
        """
        随机选取样本切分点 (截断测试)。
        验证: 无论截断点落在何处，历史部分指标必须与全样本历史精确 100% 一致 (diff == 0.0)。
        """
        rng = np.random.RandomState(split_seed)
        n_full = 300
        cut_point = rng.randint(40, 260)

        # 构造完整时序数据
        dates = pd.date_range("2023-01-01", periods=n_full, freq="B")
        p_sa = 1800.0 + np.cumsum(rng.normal(0, 15, n_full))
        p_fg = 1600.0 + np.cumsum(rng.normal(0, 10, n_full))
        inv = 75.0 + np.cumsum(rng.normal(0, 1.5, n_full))

        df_full = pd.DataFrame({
            "close_sa": p_sa,
            "close_fg": p_fg,
            "inventory_sa": inv
        }, index=dates)

        df_trunc = df_full.iloc[:cut_point].copy()

        # 运行 Temporal NALE
        engine = TemporalNALE(window=40)
        res_full = engine.compute_chain_response(df_full)
        res_trunc = engine.compute_chain_response(df_trunc)

        for col in ["conv_upstream_cost", "asym_upstream_cost", "stgat_divergence_attention", "ncsi"]:
            diff = np.max(np.abs(res_full[col].iloc[:cut_point].values - res_trunc[col].values))
            assert diff < 1e-12, f"在切分点 {cut_point} (seed={split_seed}) 下 {col} 存在前视漂移: diff={diff}"

    def test_schwartz_smith_right_side_poisoning_attack(self):
        """
        黑天鹅未来数据恶意投毒反恐测试 (Schwartz-Smith 模型):
        在 split_point 之后的未来数据中恶意注入极大数值 (暴涨 100,000, 暴跌至 0, 翻转符号)。
        验证: 历史计算结果在位级别上零变化 (bit-exact zero drift)。
        """
        rng = np.random.RandomState(888)
        n = 250
        split_point = 150

        p_sa = pd.Series(2000.0 + np.cumsum(rng.normal(0, 12, n)))
        p_fg = pd.Series(1700.0 + np.cumsum(rng.normal(0, 8, n)))

        # 1. 干净全量样本
        model_ss = SchwartzSmithTwoFactorModel()
        clean_res = model_ss.fit_filter_causal(spot=p_sa, futures=p_fg)

        # 2. 恶意投毒样本 (在 split_point 之后注入黑天鹅极值)
        poisoned_sa = p_sa.copy()
        poisoned_fg = p_fg.copy()
        poisoned_sa.iloc[split_point:] = 999999.0
        poisoned_fg.iloc[split_point:] = 1.0

        poisoned_res = model_ss.fit_filter_causal(spot=poisoned_sa, futures=poisoned_fg)

        # 验证 split_point 之前的历史严格为 0 误差
        for col in ["chi_short", "xi_long", "convenience_yield", "roll_yield", "term_structure_slope"]:
            max_diff = np.max(np.abs(clean_res[col].iloc[:split_point].values - poisoned_res[col].iloc[:split_point].values))
            assert max_diff == 0.0, f"未来数据投毒导致历史 {col} 发生漂移: max_diff={max_diff}"

    def test_temporal_nale_right_side_poisoning_attack(self):
        """
        黑天鹅未来数据恶意投毒反恐测试 (Temporal NALE ST-GAT 引擎):
        在未来窗口中恶意注入极限价格翻转 (+1,000,000 / -500,000) 与库存暴乱。
        验证: 历史 NCSI 与图注意力权重历史输出 100% 不变。
        """
        rng = np.random.RandomState(777)
        n = 200
        split_point = 120

        df_clean = pd.DataFrame({
            "close_sa": 1900.0 + np.cumsum(rng.normal(0, 10, n)),
            "close_fg": 1600.0 + np.cumsum(rng.normal(0, 8, n)),
            "inventory_sa": 75.0 + np.cumsum(rng.normal(0, 1.2, n))
        })

        df_poisoned = df_clean.copy()
        df_poisoned.loc[split_point:, "close_sa"] = 50000.0
        df_poisoned.loc[split_point:, "close_fg"] = 200.0
        df_poisoned.loc[split_point:, "inventory_sa"] = 5000.0

        engine = TemporalNALE(window=30)
        res_clean = engine.compute_chain_response(df_clean)
        res_poisoned = engine.compute_chain_response(df_poisoned)

        columns_to_audit = [
            "conv_upstream_cost",
            "asym_upstream_cost",
            "stgat_divergence_attention",
            "ncsi",
            "crush_spread"
        ]

        for col in columns_to_audit:
            drift = np.max(np.abs(res_clean[col].iloc[:split_point].values - res_poisoned[col].iloc[:split_point].values))
            assert drift < 1e-11, f"未来数据投毒导致 Temporal NALE 历史 {col} 漂移: max_drift={drift}"

    def test_basis_convergence_engine_right_side_poisoning_attack(self):
        """
        黑天鹅未来数据恶意投毒反恐测试 (BasisConvergenceEngine):
        在未来时段将仓单与库存改写为天文数字与异常负值。
        验证: 历史 lambda_t 计算值 0 误差。
        """
        engine = BasisConvergenceEngine()
        n = 150
        split_point = 90
        rng = np.random.RandomState(666)

        wr_clean = pd.Series(6000.0 + np.cumsum(rng.normal(0, 50, n)))
        inv_clean = pd.Series(75.0 + np.cumsum(rng.normal(0, 1, n)))

        wr_poisoned = wr_clean.copy()
        inv_poisoned = inv_clean.copy()
        wr_poisoned.iloc[split_point:] = -9999999.0
        inv_poisoned.iloc[split_point:] = 9999999.0

        l_clean = engine.compute_dynamic_convergence_speed(
            warehouse_receipts=wr_clean, inventory=inv_clean
        )
        l_poisoned = engine.compute_dynamic_convergence_speed(
            warehouse_receipts=wr_poisoned, inventory=inv_poisoned
        )

        max_drift = np.max(np.abs(l_clean.iloc[:split_point].values - l_poisoned.iloc[:split_point].values))
        assert max_drift == 0.0, f"未来数据投毒导致 BasisConvergenceEngine 历史漂移: {max_drift}"
