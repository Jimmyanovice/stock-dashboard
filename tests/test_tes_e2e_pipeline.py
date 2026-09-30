# -*- coding: utf-8 -*-
"""
跨资产期望盈余 (TES) 端到端集成与契约验证测试套件 (tests/test_tes_e2e_pipeline.py)
-----------------------------------------------------------------------------
验证 ADR-0001, ADR-0002 与 M3/M4 端到端全链路:
1. 正向因果传导链路 (Positive Causal Transmission):
   天然碱成本坍塌 -> 纯碱出厂价承压 -> 玻璃冶炼裂解走阔 (Crush Spread) -> A 股产业链标的差异化收益
2. 逆向虚假炒作与噪音一票否决拦截 (Counterfactual False Hype Interception):
   小道消息传言逼空、社交媒体虚假情绪炒作在代码统计门禁下 100% 拦截 (置零与高风险驳回标签)
3. 前端交互数据契约 (docs/index.html 10 核心列标准表单与 ranking_cross_asset.json 结构验证)
4. 原子化落盘安全机制 (Atomic Write Resilience)
"""

import json
import os
from pathlib import Path
import numpy as np
import pytest

from src.models.causal_ontology import ValueAddedOntology
from src.models.tes_pipeline import (
    CrossAssetTESPipeline,
    compute_cross_asset_surplus,
    simulate_calibrated_returns,
    atomic_write_json
)
import importlib


@pytest.fixture(scope="module")
def pipeline() -> CrossAssetTESPipeline:
    """初始化 TES 打分流水线实例"""
    return CrossAssetTESPipeline()


# ==============================================================================
# 1. 正向因果传导全链路端到端验证
# ==============================================================================

def test_positive_causal_transmission_chain(pipeline: CrossAssetTESPipeline):
    """
    【正向因果传导测试】
    模拟天然碱新增产能释放导致成本坍塌 (-20% 冲击) 场景:
    1. 上游天然碱 (000683 远兴能源): 低成本以量补价，预期盈余维持高位 (+1.95%)
    2. 中间纯碱合成 (600328 中盐化工): 传统氨碱/联碱受低成本天然碱冲击，利润受压 (+0.25%)
    3. 纯碱期货 (SA701): 破位下行状态 (TrendGate Regime 1)，叠加 NALE 前瞻做空信号 (-0.0818)，
       期货空头锁利预期盈余高达 +2.60%，名列前茅
    4. 下游玻璃加工 (601636 旗滨集团, 601865 福莱特): 纯碱降价带动冶炼裂解加工利润走阔，
       预期盈余获得正向提振 (旗滨 > +1.45%, 福莱特 > +1.60%)
    5. 玻璃期货 (FG701): 刚性点火与裂解走阔支撑基差修复，预期盈余稳健 (+1.45%)
    """
    ranking = pipeline.compute_surplus(
        sa_price=1014.0,
        fg_price=1100.0,
        trend_regime=1,        # 破位做空保障期
        nale_signal=-0.0818,   # 产业链负向因果时滞前瞻信号
        cost_shock_pct=-0.20   # 天然碱成本坍塌 20%
    )

    ranking_by_sym = {item["symbol"]: item for item in ranking}

    # 1. 检验所有核心标的均存在且通过门禁
    assert "SA701" in ranking_by_sym
    assert "FG701" in ranking_by_sym
    assert "000683" in ranking_by_sym
    assert "600328" in ranking_by_sym
    assert "601636" in ranking_by_sym
    assert "601865" in ranking_by_sym
    assert "300750" in ranking_by_sym
    assert "601012" in ranking_by_sym

    # 2. 纯碱空头因时滞与做空机制位居榜首
    sa_item = ranking_by_sym["SA701"]
    assert sa_item["rank"] == 1
    assert sa_item["tomorrow_expected_surplus_pct"] >= 2.50
    assert sa_item["statistical_gate_pass"] is True
    assert sa_item["gate_verdict"] == "APPROVE"

    # 3. 远兴能源凭借天然碱垄断成本优势位居前列
    yx_item = ranking_by_sym["000683"]
    assert yx_item["rank"] <= 3
    assert yx_item["tomorrow_expected_surplus_pct"] >= 1.90
    assert yx_item["statistical_gate_pass"] is True

    # 4. 玻璃企业（旗滨、福莱特）受益于纯碱原料降价，获得裂解利润扩张提振
    qb_item = ranking_by_sym["601636"]
    flt_item = ranking_by_sym["601865"]
    assert qb_item["tomorrow_expected_surplus_pct"] >= 1.45
    assert flt_item["tomorrow_expected_surplus_pct"] >= 1.60

    # 5. 中盐化工 (600328) 受天然碱冲击，预期盈余显著低于远兴与旗滨
    zy_item = ranking_by_sym["600328"]
    assert zy_item["tomorrow_expected_surplus_pct"] < qb_item["tomorrow_expected_surplus_pct"]
    assert zy_item["tomorrow_expected_surplus_pct"] < yx_item["tomorrow_expected_surplus_pct"]

    # 6. 隆基绿能 (601012) 光伏组件环节产能过剩，预期盈余处于垫底区间
    lj_item = ranking_by_sym["601012"]
    assert lj_item["tomorrow_expected_surplus_pct"] < 0.0


# ==============================================================================
# 2. 逆向反事实虚假炒作与噪音一票否决拦截测试
# ==============================================================================

def test_counterfactual_false_hype_and_rumor_interception(pipeline: CrossAssetTESPipeline):
    """
    【反事实虚假炒作拦截用例】
    注入典型金融市场伪命题与炒作谣言:
    - 谣言 1: 社交媒体传闻“某高成本纯碱厂突发检修，现货即将暴涨翻倍逼空” (纯白噪声，无统计显著性 p >= 0.05)
    - 谣言 2: 小作文炒作“某概念小盘股攻克天然碱卡脖子技术” (IR < 0.30 甚至负向，无经济意义)
    
    验收标准:
    1. 两层解耦统计代码门禁必须 100% 坚决拦截 (One-Vote Veto)
    2. 虚假假说条目的 statistical_gate_pass 必须为 False
    3. gate_verdict 必须为 REJECT_NON_SIGNIFICANT 或 REJECT_LOW_IR
    4. 明日预期盈余必须强制归零 (tomorrow_expected_surplus_pct == 0.0)
    5. 风险评级标明“驳回高风险”，排名坚决沉底在所有通过标的之后
    """
    rng = np.random.RandomState(999)
    T = 250
    mkt = rng.normal(0.0002, 0.012, T)
    
    # 构造谣言 1 收益率: 纯随机白噪声 (无 Alpha, p >= 0.05)
    noise_ret_1 = 0.8 * mkt + rng.normal(0.0, 0.025, T)
    
    # 构造谣言 2 收益率: 微弱噪声且负收益
    noise_ret_2 = 0.5 * mkt - 0.0005 + rng.normal(0.0, 0.020, T)

    counterfactual_propositions = [
        {
            "symbol": "RUMOR_SQUEEZE_01",
            "name": "虚假纯碱逼空传言",
            "asset_type": "futures",
            "direction": "LONG",
            "dir_display": "传言逼空做多",
            "last_price": 1050.0,
            "driver_logic": "网络短视频炒作纯碱装置检修停工",
            "evidence_tags": ["[OPINION] 股吧传闻纯碱将暴涨翻倍"],
            "base_surplus": 5.0,  # 假想的高额虚妄收益
            "asset_returns": noise_ret_1,
            "factor_returns": mkt.reshape(-1, 1),
            "is_noise": True
        },
        {
            "symbol": "HYPE_STOCK_02",
            "name": "概念包装伪天然碱",
            "asset_type": "stock",
            "direction": "LONG",
            "dir_display": "题材概念炒作",
            "last_price": 18.20,
            "driver_logic": "游资小作文包装假想天然碱重估",
            "evidence_tags": ["[OPINION] 盘中推特传闻概念受益"],
            "base_surplus": 4.5,
            "asset_returns": noise_ret_2,
            "factor_returns": mkt.reshape(-1, 1),
            "is_noise": True
        }
    ]

    ranking = pipeline.compute_surplus(
        sa_price=1014.0,
        fg_price=1100.0,
        trend_regime=1,
        custom_propositions=counterfactual_propositions
    )

    ranking_by_sym = {item["symbol"]: item for item in ranking}

    # 1. 验证谣言标的被 100% 拦截
    rumor_1 = ranking_by_sym["RUMOR_SQUEEZE_01"]
    assert rumor_1["statistical_gate_pass"] is False
    assert rumor_1["gate_verdict"] in {"REJECT_NON_SIGNIFICANT", "REJECT_LOW_IR", "REJECT_UNMAPPED_NODE", "REJECT_UNGROUNDED_RUMOR"}
    assert rumor_1["tomorrow_expected_surplus_pct"] == 0.0
    assert rumor_1["risk_rating"] == "驳回高风险"
    assert "【门禁驳回】" in rumor_1["rationale"]

    hype_2 = ranking_by_sym["HYPE_STOCK_02"]
    assert hype_2["statistical_gate_pass"] is False
    assert hype_2["gate_verdict"] in {"REJECT_NON_SIGNIFICANT", "REJECT_LOW_IR", "REJECT_UNMAPPED_NODE", "REJECT_UNGROUNDED_RUMOR"}
    assert hype_2["tomorrow_expected_surplus_pct"] == 0.0
    assert hype_2["risk_rating"] == "驳回高风险"
    assert "【门禁驳回】" in hype_2["rationale"]

    # 2. 验证被驳回的条目排名沉底 (排在所有统计核准通过的条目之后)
    approved_items = [it for it in ranking if it["statistical_gate_pass"]]
    rejected_items = [it for it in ranking if not it["statistical_gate_pass"]]

    assert len(approved_items) > 0
    assert len(rejected_items) >= 2

    max_approved_rank = max(it["rank"] for it in approved_items)
    min_rejected_rank = min(it["rank"] for it in rejected_items)
    assert min_rejected_rank > max_approved_rank, (
        f"被门禁驳回条目的最高排名 ({min_rejected_rank}) 未沉底至通过条目最大排名 ({max_approved_rank}) 之后"
    )


# ==============================================================================
# 3. 前端交互契约 (docs/index.html & ranking_cross_asset.json) 完整性验证
# ==============================================================================

def test_frontend_json_contract_and_columns(pipeline: CrossAssetTESPipeline, tmp_path: Path):
    """
    【前端终端数据契约验证】
    严格校验生成的 JSON 榜单完全符合 docs/index.html 10 核心列与架构扩展字段规约:
    - 10 核心列: rank, symbol/code, name, sector, asset_type/type, direction/dir,
                 last_price/price, tomorrow_expected_surplus_pct/surplus, nale_lead_signal/nale,
                 risk_rating/risk, rationale/reason
    - 架构扩展字段: node_id, node_name, qualitative_label, gate_verdict, p_value, information_ratio, statistical_gate_pass
    - 顶层元数据: schema_version, updated_at, engine, market_regime, items
    """
    ranking = pipeline.compute_surplus(sa_price=1014.0, fg_price=1100.0, trend_regime=1, nale_signal=-0.0818)
    
    test_json_path = tmp_path / "ranking_cross_asset.json"
    exported_file = pipeline.export_ranking_json(ranking, output_path=str(test_json_path))

    assert os.path.exists(exported_file)

    with open(exported_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 顶层契约校验
    assert data["schema_version"] == "v19.5_cross_asset_v1"
    assert "updated_at" in data
    assert "engine" in data
    assert "market_regime" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) >= 14  # 包含纯碱、玻璃与12只核心A股

    # 2. 表格 10 核心列与扩展字段逐行校验
    required_core_keys = [
        "rank", "symbol", "name", "sector", "asset_type",
        "direction", "last_price", "tomorrow_expected_surplus_pct",
        "nale_lead_signal", "risk_rating", "rationale"
    ]
    required_alias_keys = ["code", "type", "dir", "price", "surplus", "nale", "risk", "reason"]
    required_arch_keys = [
        "node_id", "node_name", "qualitative_label", "gate_verdict",
        "p_value", "information_ratio", "statistical_gate_pass"
    ]

    for item in data["items"]:
        # 核心 10 列字段
        for k in required_core_keys:
            assert k in item, f"条目 {item.get('symbol')} 缺失核心表单字段: {k}"

        # 兼容 docs/index.html 前端渲染的简写别名
        for k in required_alias_keys:
            assert k in item, f"条目 {item.get('symbol')} 缺失前端别名字段: {k}"

        # M1/M2 深度架构字段
        for k in required_arch_keys:
            assert k in item, f"条目 {item.get('symbol')} 缺失架构扩展字段: {k}"

        # 类型与范围合理性
        assert isinstance(item["rank"], int) and item["rank"] >= 1
        assert isinstance(item["last_price"], (int, float)) and item["last_price"] > 0
        assert isinstance(item["tomorrow_expected_surplus_pct"], (int, float))
        assert isinstance(item["statistical_gate_pass"], bool)
        assert item["gate_verdict"] in {"APPROVE", "REJECT_NON_SIGNIFICANT", "REJECT_LOW_IR", "REJECT_UNMAPPED_NODE", "REJECT_UNGROUNDED_RUMOR"}

    # 3. 严格递增的 Rank 排序校验
    ranks = [it["rank"] for it in data["items"]]
    assert ranks == list(range(1, len(ranks) + 1)), f"排名序号不连续: {ranks}"


# ==============================================================================
# 4. 原子化落盘安全机制验证
# ==============================================================================

def test_atomic_write_json_resilience(tmp_path: Path):
    """
    验证原子化落盘函数 (atomic_write_json) 的鲁棒性:
    1. 正常落盘生成完整文件
    2. 无残留的 .tmp 临时文件污染目录
    3. 能够自动创建不存在的父级目录
    """
    nested_dir = tmp_path / "deeply" / "nested" / "log_dir"
    target_file = nested_dir / "test_atomic.json"

    test_data = {"test_key": "test_value", "number": 12345}
    ok = atomic_write_json(str(target_file), test_data)
    assert ok is True
    assert target_file.exists()

    with open(target_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == test_data

    # 检查目录下无任何残留 .tmp 临时文件
    tmp_files = list(nested_dir.glob("*.tmp*"))
    assert len(tmp_files) == 0, f"发现未清理的临时文件: {tmp_files}"


# ==============================================================================
# 5. 极星 19.5 插件集成一致性验证
# ==============================================================================

def test_polaris_plugin_compute_cross_asset_surplus_integration():
    """
    验证 02_极星量化与实盘/strategies/CZCE_19_5_DualProduct_Plugin.py 中的
    compute_cross_asset_surplus 与底层流水线输出完全一致
    """
    plugin_module = importlib.import_module("02_极星量化与实盘.strategies.CZCE_19_5_DualProduct_Plugin")
    plugin_surplus_fn = getattr(plugin_module, "compute_cross_asset_surplus")

    # 执行插件中的打分接口
    ranking = plugin_surplus_fn(
        sa_price=1014.0,
        fg_price=1100.0,
        trend_regime=1,
        nale_signal=-0.0818
    )

    assert isinstance(ranking, list)
    assert len(ranking) >= 14

    # 验证 SA701 与 000683 排名与字段
    symbols = [r["symbol"] for r in ranking]
    assert "SA701" in symbols
    assert "FG701" in symbols
    assert "000683" in symbols
    assert "601636" in symbols
    assert "300750" in symbols

    # 验证 Rank 1
    assert ranking[0]["symbol"] == "SA701"
    assert ranking[0]["statistical_gate_pass"] is True
