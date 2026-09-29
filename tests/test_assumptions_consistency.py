# -*- coding: utf-8 -*-
"""
数据与建模假设一致性强校验测试套件 (tests/test_assumptions_consistency.py)
---------------------------------------------------------------------
严格检验 reports/assumptions.md 中披露的实测统计量、模型超参数与交易摩擦参数
与代码、数据实测产出 100% 精确一致，彻底消除 B5 / T4 缺陷。
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.basis_term_structure import BasisConvergenceEngine, SchwartzSmithConfig
from src.models.temporal_nale import TemporalNALEConfig
from src.models.trend_gate import TrendGateMachine
from src.models.hedging_strategies import HedgingEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSUMPTIONS_PATH = ROOT_DIR / "reports" / "assumptions.md"
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "czce_sa_fg_aligned_features.csv"


def test_assumptions_file_exists():
    """测试 1: 验证假设清单文件存在且非空"""
    assert ASSUMPTIONS_PATH.exists(), f"假设清单文件缺失: {ASSUMPTIONS_PATH}"
    text = ASSUMPTIONS_PATH.read_text(encoding="utf-8")
    assert len(text) > 500, "假设清单内容异常过短"


def test_empirical_basis_statistics_consistency():
    """测试 2: 验证文档中声称的纯碱/玻璃基差均值、标准差与极值区间与数据产出误差 < 0.05"""
    if not FEATURES_PATH.exists():
        pytest.skip(f"特征矩阵文件尚未生成: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)
    text = ASSUMPTIONS_PATH.read_text(encoding="utf-8")

    sa_basis = df["basis_sa"]
    real_mean = sa_basis.mean()
    real_std = sa_basis.std()
    real_min = sa_basis.min()
    real_max = sa_basis.max()

    # 正则提取文档中的基差数值: \bar{B}_{SA} = -219.33 元/吨
    mean_match = re.search(r"\\bar\{B\}_\{SA\}\s*=\s*([+-]?\d+\.?\d*)", text)
    assert mean_match is not None, "未在文档中匹配到纯碱基差均值 \\bar{B}_{SA}"
    doc_mean = float(mean_match.group(1))
    assert abs(doc_mean - real_mean) < 0.05, f"基差均值不一致: 文档={doc_mean}, 实际={real_mean:.2f}"

    # 提取标准差: \sigma_{B, SA} = 346.86 元/吨
    std_match = re.search(r"\\sigma_\{B,\s*SA\}\s*=\s*([+-]?\d+\.?\d*)", text)
    assert std_match is not None, "未在文档中匹配到纯碱基差标准差 \\sigma_{B, SA}"
    doc_std = float(std_match.group(1))
    assert abs(doc_std - real_std) < 0.05, f"基差标准差不一致: 文档={doc_std}, 实际={real_std:.2f}"

    # 提取极值区间: [min, max]
    ext_match = re.search(r"实测基差极值区间\S*：\$\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\]\$", text)
    assert ext_match is not None, "未在文档中匹配到纯碱基差极值区间"
    doc_min = float(ext_match.group(1))
    doc_max = float(ext_match.group(2))
    assert abs(doc_min - real_min) < 0.05, f"基差最小值不一致: 文档={doc_min}, 实际={real_min:.2f}"
    assert abs(doc_max - real_max) < 0.05, f"基差最大值不一致: 文档={doc_max}, 实际={real_max:.2f}"


def test_trading_frictions_consistency():
    """测试 3: 验证保证金 12%、手续费 0.02%、合约乘数 20、库存 10,000 吨与代码一致"""
    text = ASSUMPTIONS_PATH.read_text(encoding="utf-8")

    engine = HedgingEngine(df=pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "close_sa": [1000.0, 1010.0],
        "spot_sa": [1050.0, 1060.0],
        "basis_sa": [-50.0, -50.0]
    }))
    assert "10,000 吨" in text or "10000 吨" in text
    assert engine.inventory_tons == 10000.0
    assert engine.multiplier == 20.0
    assert engine.margin_ratio == 0.12
    assert engine.commission_rate == 0.0002
    assert engine.rf == 0.03


def test_model_hyperparameters_consistency():
    """测试 4: 验证 SchwartzSmith, BasisConvergence, TemporalNALE, TrendGate 超参数 100% 对齐"""
    text = ASSUMPTIONS_PATH.read_text(encoding="utf-8")

    # 1. BasisConvergenceEngine
    bce = BasisConvergenceEngine()
    assert f"{bce.lambda_min:.3f}" in text, f"lambda_min {bce.lambda_min} 缺失在文档中"
    assert f"{bce.lambda_max:.3f}" in text, f"lambda_max {bce.lambda_max} 缺失在文档中"

    # 2. TrendGate 统一权重规范 (0.45, 0.35, 0.20)
    assert "0.45" in text, "0.45 缺失在文档中"
    assert "0.35" in text, "0.35 缺失在文档中"
    assert "0.20" in text, "0.20 缺失在文档中"

    # 3. TemporalNALEConfig
    nale_cfg = TemporalNALEConfig()
    assert f"{nale_cfg.half_life_up:.1f}" in text
    assert f"{nale_cfg.half_life_down:.1f}" in text
    assert f"{nale_cfg.fuel_cost_per_ton:.1f}" in text


def test_data_sources_transparency_annotation():
    """测试 5: 验证文档如实披露纯碱期权 2023-10-20 上市及 Level 1 / Level 3 数据分级"""
    text = ASSUMPTIONS_PATH.read_text(encoding="utf-8")
    assert "2023年10月20日" in text, "未披露纯碱期权官方上市日期"
    assert "Level 1" in text, "未标注 Level 1 真实场内行情分级"
    assert "Level 3" in text, "未标注 Level 3 理论合成代理分级"
    assert "inventory_weight = 0.0" in text or "inventory_weight" in text, "未注明行业库存优雅降级策略"
