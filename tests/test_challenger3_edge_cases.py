# -*- coding: utf-8 -*-
"""
Challenger 3 Adversarial Edge-Case Stress Suite
================================================
Empirical test harness probing:
1. Whitespace and formatting alpha bypass attempts (BUG-ENG-01 probes)
2. Whitespace symbol bypass (BUG-ONT-01 probes)
3. Word boundary collisions: "SAND", "SAFE", "SAMPLE" vs "SA", and "S" vs "SALT" (BUG-ONT-03 probes)
4. Integer and non-string symbol types without crash (BUG-ONT-02 probes)
5. Shock math propagation, sign integrity, and crush spread boundary behavior (BUG-ONT-04/05 probes)
"""

import math
import pytest
import numpy as np

from src.models.causal_ontology import (
    ValueAddedOntology,
    ProcessNodeType,
    ProcessNode,
    validate_no_numerical_alpha
)
from src.models.two_layer_engine import (
    QualitativeProposition,
    DecoupledTwoLayerEngine,
    GateVerdict,
    compute_newey_west_bandwidth
)
from src.models.tes_pipeline import (
    CrossAssetTESPipeline,
    compute_cross_asset_surplus,
    _clean_price
)


@pytest.fixture(scope="module")
def ontology() -> ValueAddedOntology:
    return ValueAddedOntology()


@pytest.fixture(scope="module")
def engine() -> DecoupledTwoLayerEngine:
    return DecoupledTwoLayerEngine(p_threshold=0.05, ir_threshold=0.30)


# ==============================================================================
# 1. Whitespace & Formatting Alpha Bypass Probes (BUG-ENG-01)
# ==============================================================================

class TestWhitespaceAlphaBypass:
    """Probing variations of numerical alpha injection bypassing regex filters."""

    @pytest.mark.parametrize("payload", [
        # Whitespace variations around '='
        "天然碱投产冲击 ALPHA=0.08",
        "天然碱投产冲击 ALPHA = 0.08",
        "天然碱投产冲击 ALPHA  =  0.08",
        "天然碱投产冲击 ALPHA\t=\t0.08",
        "天然碱投产冲击 ALPHA\n=\n0.08",
        "天然碱投产冲击 ALPHA == 0.08",
        "天然碱投产冲击 ALPHA==0.08",
        # Whitespace variations around ':'
        "天然碱投产冲击 ALPHA:0.08",
        "天然碱投产冲击 ALPHA : 0.08",
        "天然碱投产冲击 ALPHA   :   0.08",
        "天然碱投产冲击 ALPHA\t:\t0.08",
        # Case variations
        "天然碱投产冲击 alpha = 0.08",
        "天然碱投产冲击 Alpha : 0.08",
        "天然碱投产冲击 aLpHa = 0.08",
        # Predicted return variations
        "纯碱走势研判 PREDICTED_RETURN=+12%",
        "纯碱走势研判 PREDICTED RETURN: +12%",
        "纯碱走势研判 PREDICTED   RETURN : +12%",
        "纯碱走势研判 predicted return: 12%",
        "纯碱走势研判 predicted_return = 12%",
        # Target return variations
        "目标收益估算 TARGET_RETURN_PCT: 8%",
        "目标收益估算 TARGET RETURN: +8%",
        "目标收益估算 TARGET  RETURN = 8%",
        "目标收益估算 target return: 8%",
        # Expected return variations
        "预期超额估算 EXPECTED_RETURN: 10%",
        "预期超额估算 EXPECTED RETURN: +10%",
        "预期超额估算 EXPECTED  RETURN = +10%",
        "预期超额估算 expected return: 10%",
        # Confidence probability variations
        "置信概率估算 CONFIDENCE_PROBABILITY=0.95",
        "置信概率估算 CONFIDENCE PROBABILITY: 0.95",
        "置信概率估算 CONFIDENCE  PROBABILITY = 0.95",
        "置信概率估算 confidence probability: 0.95",
    ])
    def test_alpha_bypass_in_driver_logic_rejected(self, payload: str):
        """Verify that any whitespace/case permutation of forbidden alpha raises ValueError."""
        with pytest.raises(ValueError, match="Integrity Violation"):
            QualitativeProposition(
                proposition_id="PROP_EVASION",
                symbol="SA701",
                node_id="NODE_03_SODA_SYN",
                driver_logic=payload,
                direction="LONG",
                evidence_tags=["[FACT] 基础事实"]
            )

    @pytest.mark.parametrize("payload", [
        "研报摘要: ALPHA = 0.15",
        "PREDICTED RETURN: 10%",
        "EXPECTED RETURN = 5%"
    ])
    def test_alpha_bypass_in_source_title_and_evidence_tags_rejected(self, payload: str):
        """Verify injection in source_title or evidence_tags is also blocked."""
        # Injection in source_title
        with pytest.raises(ValueError, match="Integrity Violation"):
            QualitativeProposition(
                proposition_id="PROP_EVASION_TITLE",
                symbol="SA701",
                node_id="NODE_03_SODA_SYN",
                driver_logic="正常基本面逻辑",
                direction="LONG",
                source_title=payload,
                evidence_tags=["[FACT] 基础事实"]
            )

        # Injection in evidence_tags
        with pytest.raises(ValueError, match="Integrity Violation"):
            QualitativeProposition(
                proposition_id="PROP_EVASION_TAG",
                symbol="SA701",
                node_id="NODE_03_SODA_SYN",
                driver_logic="正常基本面逻辑",
                direction="LONG",
                evidence_tags=["[FACT] 基础事实", payload]
            )

    def test_legitimate_qualitative_phrasing_not_false_alarmed(self):
        """Verify legitimate phrasing containing the words alpha, return, confidence are NOT falsely blocked."""
        legit_cases = [
            "关注公司长期Alpha超额收益能力的培育",
            "在行业低谷期探索特质Alpha的来源",
            "下游终端组件对光伏玻璃的需求正在逐步return回暖",
            "市场各方对天然碱新增产能达产具有较高confidence信心",
            "根据历史统计规律，行业呈现强烈的周期特征"
        ]
        for text in legit_cases:
            prop = QualitativeProposition(
                proposition_id="PROP_LEGIT",
                symbol="SA701",
                node_id="NODE_03_SODA_SYN",
                driver_logic=text,
                direction="LONG",
                evidence_tags=["[FACT] 真实产能数据"]
            )
            assert prop.driver_logic == text


# ==============================================================================
# 2. Whitespace Symbol Bypass Probes (BUG-ONT-01)
# ==============================================================================

class TestWhitespaceSymbolBypass:
    """Probing empty and whitespace inputs against ontology lookup and validation."""

    @pytest.mark.parametrize("blank", [
        "",
        " ",
        "   ",
        "\t",
        "\n",
        "\r\n",
        "  \t \n  \r\n  ",
        None
    ])
    def test_whitespace_symbol_never_resolves_to_node(self, ontology: ValueAddedOntology, blank):
        """Whitespace strings or None must always return None."""
        assert ontology.get_node_by_symbol(blank) is None

    @pytest.mark.parametrize("blank", [
        "",
        " ",
        "   ",
        "\t",
        "\n",
        None
    ])
    @pytest.mark.parametrize("level", [1, 2, 3, 4])
    def test_whitespace_symbol_never_validates_as_any_level(self, ontology: ValueAddedOntology, blank, level):
        """Whitespace or None must fail asset mapping validation for any claimed level."""
        assert ontology.validate_asset_mapping(blank, claimed_level=level) is False

    @pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n", None])
    def test_process_node_contains_symbol_with_blanks(self, ontology: ValueAddedOntology, blank):
        """All process nodes must return False for blank symbols."""
        for node in ontology.all_nodes().values():
            assert node.contains_symbol(blank) is False


# ==============================================================================
# 3. Word Boundary Collision Probes (BUG-ONT-03)
# ==============================================================================

class TestWordBoundaryCollision:
    """Probing exact boundaries: 'SAND', 'SAFE', 'SAMPLE' vs 'SA', 'S' vs 'SALT'."""

    @pytest.mark.parametrize("word", [
        "SAND",       # Silica sand (glass feedstock)
        "SAMPLE",     # Sampling token
        "SAFE",       # Safety glass keyword
        "SALARY",     # Unrelated word
        "SAD",        # Unrelated word
        "S",          # Single letter 'S'
        "SA_",        # Malformed prefix
        "SA_INVALID", # Invalid futures contract
        "SA-701",     # Invalid delimiter
        "SA 701",     # Invalid space in symbol
    ])
    def test_sa_collision_words_do_not_match_soda_futures(self, ontology: ValueAddedOntology, word: str):
        """Non-contract words starting with SA or S must never resolve to NODE_03_SODA_SYN or NODE_02_SALT."""
        node = ontology.get_node_by_symbol(word)
        assert node is None, f"Word '{word}' erroneously resolved to {node.node_id if node else None}"

    @pytest.mark.parametrize("word", [
        "F",
        "FG_",
        "FG_SAMPLE",
        "FIGHT",
        "FLAG",
        "FORWARD",
        "FG-701",
        "FG 701"
    ])
    def test_fg_collision_words_do_not_match_glass_futures(self, ontology: ValueAddedOntology, word: str):
        """Non-contract words starting with FG or F must never resolve to NODE_04_FLOAT_GLASS."""
        node = ontology.get_node_by_symbol(word)
        assert node is None, f"Word '{word}' erroneously resolved to {node.node_id if node else None}"

    @pytest.mark.parametrize("valid_contract,expected_node_id", [
        ("SA", "NODE_03_SODA_SYN"),
        ("SA0", "NODE_03_SODA_SYN"),
        ("SA701", "NODE_03_SODA_SYN"),
        ("SA509", "NODE_03_SODA_SYN"),
        ("FG", "NODE_04_FLOAT_GLASS"),
        ("FG0", "NODE_04_FLOAT_GLASS"),
        ("FG701", "NODE_04_FLOAT_GLASS"),
        ("FG509", "NODE_04_FLOAT_GLASS"),
        ("TRONA_ORE", "NODE_01_TRONA"),
        ("SALT_RAW", "NODE_02_SALT"),
    ])
    def test_valid_futures_and_commodities_resolve_correctly(
        self,
        ontology: ValueAddedOntology,
        valid_contract: str,
        expected_node_id: str
    ):
        """Valid futures contract codes with digits must match correctly."""
        node = ontology.get_node_by_symbol(valid_contract)
        assert node is not None, f"Valid commodity {valid_contract} failed to resolve"
        assert node.node_id == expected_node_id


# ==============================================================================
# 4. Integer and Non-String Symbol Probes (BUG-ONT-02)
# ==============================================================================

class TestIntegerAndNonStringSymbols:
    """Probing integer stock codes and diverse non-string data types."""

    @pytest.mark.parametrize("code_int,expected_node_id", [
        (600328, "NODE_03_SODA_SYN"),      # 中盐化工
        (600409, "NODE_03_SODA_SYN"),      # 三友化工
        (603077, "NODE_03_SODA_SYN"),      # 和邦生物
        (601636, "NODE_04_FLOAT_GLASS"),   # 旗滨集团
        (600586, "NODE_04_FLOAT_GLASS"),   # 金晶科技
        (601865, "NODE_05_PV_GLASS"),      # 福莱特
        (600660, "NODE_06_AUTO_GLASS"),    # 福耀玻璃
        (601012, "NODE_07_END_MODULE"),    # 隆基绿能
        (600438, "NODE_07_END_MODULE"),    # 通威股份
        (300750, "NODE_08_END_STORAGE"),   # 宁德时代
        (603299, "NODE_02_SALT"),          # 苏盐井神
    ])
    def test_integer_stock_codes_resolve_without_error(
        self,
        ontology: ValueAddedOntology,
        code_int: int,
        expected_node_id: str
    ):
        """Integer stock codes must resolve correctly without raising AttributeError."""
        node = ontology.get_node_by_symbol(code_int)
        assert node is not None, f"Integer code {code_int} failed to resolve"
        assert node.node_id == expected_node_id

    @pytest.mark.parametrize("invalid_type_input", [
        0,
        -1,
        999999,
        3.14159,
        True,
        False,
        ["SA701"],
        {"code": "600328"}
    ])
    def test_exotic_non_string_types_handled_gracefully(
        self,
        ontology: ValueAddedOntology,
        invalid_type_input
    ):
        """Non-string/exotic inputs must return None or False without crashing."""
        res_node = ontology.get_node_by_symbol(invalid_type_input)
        # 0, -1, 999999, 3.14159, True, False, etc. should safely return None (or match if stringified matches)
        if isinstance(invalid_type_input, (int, float)) and invalid_type_input not in [600328, 601636, 601865, 300750, 601012, 600438, 600586, 600660, 603077, 600409, 603299]:
            assert res_node is None

        res_mapping = ontology.validate_asset_mapping(invalid_type_input, claimed_level=1)
        assert isinstance(res_mapping, bool)


# ==============================================================================
# 5. Shock Math & Causal Continuity Probes (BUG-ONT-04 / BUG-ONT-05)
# ==============================================================================

class TestShockMathAndCausalContinuity:
    """Probing shock propagation math, sign consistency, and crush spread boundary physics."""

    def test_shock_propagation_exact_math(self, ontology: ValueAddedOntology):
        """
        Verify mathematical formulation of propagate_cost_shock:
        Source: NODE_01_TRONA, shock = -0.20
        L1 (TRONA): -0.2000
        L2 (SODA_SYN): -0.20 * 0.85 = -0.1700
        L3 (FLOAT_GLASS): -(-0.1700 * 0.20 * 1.5) = +0.0510
        L3 (PV_GLASS): -(-0.1700 * 0.20 * 1.5) = +0.0510
        L3 (AUTO_GLASS): 0.0510 * 0.65 = +0.0332
        L4 (END_MODULE): 0.0510 * 0.65 = +0.0332
        L4 (END_STORAGE): 0.03315 * 0.65 = +0.0215
        """
        impacts = ontology.propagate_cost_shock("NODE_01_TRONA", shock_pct=-0.20, damping_factor=0.65)
        
        assert impacts["NODE_01_TRONA"] == -0.2000
        assert impacts["NODE_03_SODA_SYN"] == -0.1700
        assert impacts["NODE_04_FLOAT_GLASS"] == 0.0510
        assert impacts["NODE_05_PV_GLASS"] == 0.0510
        # IEEE 754: 0.051 * 0.65 = 0.033149999999999996, rounded to 4 decimals is 0.0331
        assert impacts["NODE_06_AUTO_GLASS"] == round(0.0510 * 0.65, 4)
        assert impacts["NODE_07_END_MODULE"] == round(0.0510 * 0.65, 4)
        assert impacts["NODE_08_END_STORAGE"] == round(impacts["NODE_06_AUTO_GLASS"] * 0.65, 4)

        # All glass and downstream terminal modules must be strictly positive (margin expansion)
        for nid in ["NODE_04_FLOAT_GLASS", "NODE_05_PV_GLASS", "NODE_06_AUTO_GLASS", "NODE_07_END_MODULE", "NODE_08_END_STORAGE"]:
            assert impacts[nid] > 0, f"Node {nid} has non-positive shock: {impacts[nid]}"

    def test_unregistered_node_shock_raises_key_error(self, ontology: ValueAddedOntology):
        """Propagating shock from an unmapped node must raise KeyError."""
        with pytest.raises(KeyError, match="not found in ontology"):
            ontology.propagate_cost_shock("UNKNOWN_GHOST_NODE", shock_pct=-0.20)

    def test_crush_spread_boundary_physics(self, ontology: ValueAddedOntology):
        """
        Verify crush spread under extreme market conditions:
        CrushSpread = P_FG - (0.20 * P_SA) - 350
        """
        # Baseline
        base_cs = ontology.compute_crush_spread(fg_price=1100.0, sa_price=1014.0)
        assert round(base_cs, 1) == 547.2

        # Inverted / Negative Crush Spread (High SA, low FG)
        inv_cs = ontology.compute_crush_spread(fg_price=800.0, sa_price=3000.0)
        # 800 - 0.20 * 3000 - 350 = 800 - 600 - 350 = -150.0
        assert round(inv_cs, 1) == -150.0

        # Zero price boundary
        zero_cs = ontology.compute_crush_spread(fg_price=0.0, sa_price=0.0, fuel_cost=350.0)
        assert round(zero_cs, 1) == -350.0


# ==============================================================================
# 6. Pipeline NaN / Extreme Input Sanitation
# ==============================================================================

class TestPipelinePriceSanitation:
    """Probing _clean_price and pipeline reaction to NaN, Inf, negative, zero."""

    @pytest.mark.parametrize("bad_val,fallback,expected", [
        (float("nan"), 1014.0, 1014.0),
        (float("inf"), 1014.0, 1014.0),
        (float("-inf"), 1014.0, 1014.0),
        (0.0, 1014.0, 1014.0),
        (-100.0, 1014.0, 1014.0),
        (None, 1014.0, 1014.0),
        ("not_a_number", 1014.0, 1014.0),
        (1234.5, 1014.0, 1234.5),
    ])
    def test_clean_price_sanitation(self, bad_val, fallback, expected):
        """_clean_price must sanitize all corrupt numeric values to fallback."""
        assert _clean_price(bad_val, fallback=fallback) == expected
