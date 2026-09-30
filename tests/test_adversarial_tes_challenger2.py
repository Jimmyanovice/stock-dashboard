# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite - Challenger 2
============================================
Comprehensive test harness covering all 4 mission requirements:
1. Pure Gaussian noise returns & ungrounded rumors 100% rejection by code gate.
2. Positive transmission monotonicity: Trona cost drop expands downstream float/PV glass margins and penalizes synthetic soda ash margins.
3. Validate ranking_cross_asset.json: all 10 frontend table columns exist, datatypes match, values are valid floats/strings, ranking order strictly descending.
4. Atomic write safety: concurrent read/write and interrupt simulation.
5. Robustness to non-finite inputs (NaN/Inf) causing RFC 8259 JSON corruption.
"""

import os
import sys
import json
import time
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pytest

from src.models.causal_ontology import ValueAddedOntology, ProcessNodeType
from src.models.two_layer_engine import DecoupledTwoLayerEngine, QualitativeProposition
from src.models.tes_pipeline import (
    CrossAssetTESPipeline,
    compute_cross_asset_surplus,
    simulate_calibrated_returns,
    atomic_write_json,
    DEFAULT_RANKING_FILE
)


# ==============================================================================
# Challenge 1A: Pure Gaussian Noise Interception Rate
# ==============================================================================

def test_pure_gaussian_noise_interception_rate():
    """
    Adversarial Challenge 1A:
    Feed pure Gaussian noise returns (drift=0, beta*mkt + eps, alpha=0).
    Verify whether 100% of false hype is rejected by the code gate.
    """
    engine = DecoupledTwoLayerEngine(p_threshold=0.05, ir_threshold=0.30)
    
    total_trials = 500
    leaks = []
    
    for i in range(total_trials):
        rng = np.random.RandomState(42000 + i)
        T = rng.randint(40, 500)
        mkt = rng.normal(0.0002, 0.012, T)
        
        beta = rng.uniform(0.2, 1.5)
        sigma = rng.uniform(0.01, 0.04)
        eps = rng.normal(0.0, sigma, T)
        asset_ret = beta * mkt + eps
        
        direction = "LONG" if (i % 2 == 0) else "SHORT"
        prop = QualitativeProposition(
            proposition_id=f"PROP_NOISE_{i}",
            symbol=f"RUMOR_{i}",
            node_id="UNKNOWN_NODE",
            driver_logic="社交媒体纯谣言与概念小作文",
            direction=direction,
            evidence_tags=["[OPINION] 股吧传闻", "[OPINION] 盘中推特小作文"]
        )
        
        verdict = engine.verify_proposition(prop, asset_ret, mkt.reshape(-1, 1), annualize=True)
        if verdict.passed:
            leaks.append((i, T, direction, verdict.alpha_hat, verdict.p_value, verdict.information_ratio))
            
    leak_rate = len(leaks) / total_trials
    print(f"\n[Challenge 1A] Pure Gaussian Noise Trials: {total_trials}, Leaks: {len(leaks)} ({leak_rate*100:.2f}%)")
    for leak in leaks[:5]:
        print(f"  Leak detail: Trial {leak[0]}, T={leak[1]}, dir={leak[2]}, alpha={leak[3]:.4f}, p={leak[4]:.4f}, IR={leak[5]:.4f}")
        
    assert len(leaks) == 0, (
        f"CRITICAL: Code gate leaked {len(leaks)}/{total_trials} ({leak_rate*100:.2f}%) "
        f"pure Gaussian noise false hype cases! Fails 100% false hype rejection requirement."
    )


# ==============================================================================
# Challenge 1B: Ungrounded Social Media Rumors Rejection
# ==============================================================================

def test_ungrounded_social_media_rumor_rejection():
    """
    Adversarial Challenge 1B:
    Ungrounded social media rumors (symbols that do NOT exist in the causal ontology,
    or propositions having only [OPINION] / [UNVERIFIED] tags without any [FACT] tag).
    Verify whether the pipeline admits ungrounded rumors with spurious returns.
    """
    pipeline = CrossAssetTESPipeline(p_threshold=0.05, ir_threshold=0.30)
    
    rumors = []
    for i in range(20):
        rng = np.random.RandomState(8888 + i)
        T = 250
        mkt = rng.normal(0.0002, 0.012, T)
        drift = 0.0018
        eps = rng.normal(0.0, 0.005, T)
        asset_ret = drift + 1.05 * mkt + eps
        
        rumors.append({
            "symbol": f"UNGROUNDED_MEME_{i:02d}",
            "name": f"虚构小盘概念股_{i:02d}",
            "asset_type": "stock",
            "direction": "LONG",
            "dir_display": "题材概念炒作",
            "last_price": 12.50,
            "driver_logic": "游资社群发帖声称该股攻克天然碱低成本提取核心技术",
            "evidence_tags": ["[OPINION] 微信群聊天记录", "[OPINION] 股吧小作文"],
            "base_surplus": 3.80,
            "asset_returns": asset_ret,
            "factor_returns": mkt.reshape(-1, 1),
            "is_noise": False
        })
        
    ranking = pipeline.compute_surplus(custom_propositions=rumors)
    ranking_by_sym = {x["symbol"]: x for x in ranking}
    
    admitted = []
    for i in range(20):
        sym = f"UNGROUNDED_MEME_{i:02d}"
        item = ranking_by_sym[sym]
        if item["statistical_gate_pass"]:
            admitted.append((sym, item["node_id"], item["rank"], item["tomorrow_expected_surplus_pct"]))
            
    print(f"\n[Challenge 1B] Ungrounded Social Media Rumors Admitted: {len(admitted)}/20")
    for adm in admitted[:5]:
        print(f"  Admitted: {adm[0]}, node={adm[1]}, rank={adm[2]}, surplus={adm[3]}%")
        
    assert len(admitted) == 0, (
        f"CRITICAL: Pipeline admitted {len(admitted)} ungrounded social media rumors into ranking! "
        f"Lacks ontology presence and [FACT] evidence verification."
    )


# ==============================================================================
# Challenge 2: Positive Transmission Monotonicity
# ==============================================================================

def test_positive_transmission_monotonicity_adversarial():
    """
    Adversarial Challenge 2:
    Verify that a massive Trona cost drop consistently:
    1. Expands downstream float glass margins (NODE_04_FLOAT_GLASS / 601636 旗滨集团)
    2. Expands downstream PV glass margins (NODE_05_PV_GLASS / 601865 福莱特)
    3. Penalizes synthetic soda ash margins (NODE_03_SODA_SYN / 600328 中盐化工, 000822 山东海化, 600409 三友化工)
    """
    pipeline = CrossAssetTESPipeline()
    ont = pipeline.ontology
    
    shocks = [-0.05, -0.10, -0.20, -0.30, -0.40, -0.50, -0.60]
    
    # 1. Ontology Level Verification
    print("\n[Challenge 2A] Causal Ontology Shock Propagation Monotonicity:")
    for i in range(len(shocks) - 1):
        s_curr = shocks[i]
        s_next = shocks[i+1]
        
        imp_curr = ont.propagate_cost_shock("NODE_01_TRONA", shock_pct=s_curr)
        imp_next = ont.propagate_cost_shock("NODE_01_TRONA", shock_pct=s_next)
        
        fg_curr = imp_curr.get("NODE_04_FLOAT_GLASS", 0.0)
        fg_next = imp_next.get("NODE_04_FLOAT_GLASS", 0.0)
        assert fg_next > fg_curr, f"Ontology Float glass margin failed to expand: {fg_next} <= {fg_curr}"
        
        pv_curr = imp_curr.get("NODE_05_PV_GLASS", 0.0)
        pv_next = imp_next.get("NODE_05_PV_GLASS", 0.0)
        assert pv_next > pv_curr, f"Ontology PV glass margin failed to expand: {pv_next} <= {pv_curr}"
        
        sa_curr = imp_curr.get("NODE_03_SODA_SYN", 0.0)
        sa_next = imp_next.get("NODE_03_SODA_SYN", 0.0)
        assert sa_next < sa_curr, f"Ontology Synthetic soda margin failed to penalize: {sa_next} >= {sa_curr}"
        
    print("  -> Ontology propagation passes mathematical monotonicity!")
    
    # 2. Pipeline Level Verification (CrossAssetTESPipeline.compute_surplus)
    print("\n[Challenge 2B] Pipeline compute_surplus End-to-End Monotonicity:")
    surpluses_by_shock = {}
    for s in shocks:
        res = pipeline.compute_surplus(cost_shock_pct=s)
        surpluses_by_shock[s] = {x["symbol"]: x["tomorrow_expected_surplus_pct"] for x in res}
        
    for i in range(len(shocks) - 1):
        s_curr = shocks[i]
        s_next = shocks[i+1]
        
        m_curr = surpluses_by_shock[s_curr]
        m_next = surpluses_by_shock[s_next]
        
        assert m_next["601636"] >= m_curr["601636"], (
            f"Pipeline 601636 (Float Glass) failed to expand: {m_next['601636']} < {m_curr['601636']}"
        )
        assert m_next["601865"] >= m_curr["601865"], (
            f"Pipeline 601865 (PV Glass) failed to expand: {m_next['601865']} < {m_curr['601865']}"
        )
        
        print(f"  Shock {s_curr} -> {s_next}:")
        print(f"    601636 (Float Glass)   : {m_curr['601636']} -> {m_next['601636']} (diff: {m_next['601636'] - m_curr['601636']:+.2f})")
        print(f"    601865 (PV Glass)      : {m_curr['601865']} -> {m_next['601865']} (diff: {m_next['601865'] - m_curr['601865']:+.2f})")
        print(f"    600328 (Synthetic Soda): {m_curr['600328']} -> {m_next['600328']} (diff: {m_next['600328'] - m_curr['600328']:+.2f})")
        
        assert m_next["600328"] < m_curr["600328"], (
            f"HIGH: Pipeline 600328 (Synthetic Soda) failed to be penalized by Trona cost drop: "
            f"Shock {s_next} surplus is {m_next['600328']} which is NOT strictly less than shock {s_curr} surplus {m_curr['600328']}!"
        )


# ==============================================================================
# Challenge 3: Validate ranking_cross_asset.json Frontend Contract
# ==============================================================================

def test_validate_ranking_cross_asset_json_contract():
    """
    Adversarial Challenge 3:
    Validate ranking_cross_asset.json:
    - All 10 frontend table columns exist in docs/index.html
    - Datatypes match
    - Values are valid floats/strings (no NaN, Inf, empty)
    - Ranking order is strictly descending
    """
    assert os.path.exists(DEFAULT_RANKING_FILE), f"Ranking file does not exist: {DEFAULT_RANKING_FILE}"
    
    with open(DEFAULT_RANKING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "schema_version" in data
    assert "updated_at" in data
    assert "engine" in data
    assert "market_regime" in data
    assert "items" in data
    
    items = data["items"]
    assert isinstance(items, list) and len(items) > 0, "Items list is empty"
    
    required_cols = [
        "rank", "symbol", "code", "name", "sector", "direction", "dir",
        "last_price", "price", "tomorrow_expected_surplus_pct", "surplus",
        "nale_lead_signal", "nale", "risk_rating", "risk", "rationale", "reason"
    ]
    
    for idx, item in enumerate(items):
        for col in required_cols:
            assert col in item, f"Item index {idx} ({item.get('symbol')}) missing required column '{col}'"
            
        assert isinstance(item["rank"], int), f"Rank is not int: {type(item['rank'])}"
        assert item["rank"] == idx + 1, f"Rank {item['rank']} does not match 1-based index {idx + 1}"
        assert isinstance(item["symbol"], str) and len(item["symbol"]) > 0
        assert isinstance(item["code"], str) and len(item["code"]) > 0
        assert isinstance(item["name"], str) and len(item["name"]) > 0
        assert isinstance(item["sector"], str) and len(item["sector"]) > 0
        assert isinstance(item["direction"], str) and len(item["direction"]) > 0
        assert isinstance(item["dir"], str) and len(item["dir"]) > 0
        
        p = item["last_price"]
        assert isinstance(p, (int, float)) and not math.isnan(p) and not math.isinf(p) and p > 0, f"Invalid last_price: {p}"
        
        s = item["tomorrow_expected_surplus_pct"]
        assert isinstance(s, (int, float)) and not math.isnan(s) and not math.isinf(s), f"Invalid surplus: {s}"
        assert isinstance(item["surplus"], str) and ("%" in item["surplus"])
        
        n = item["nale_lead_signal"]
        assert isinstance(n, (int, float)) and not math.isnan(n) and not math.isinf(n), f"Invalid nale: {n}"
        assert isinstance(item["nale"], str)
        
        assert isinstance(item["risk_rating"], str) and len(item["risk_rating"]) > 0
        assert isinstance(item["risk"], str) and len(item["risk"]) > 0
        assert isinstance(item["rationale"], str) and len(item["rationale"]) > 0
        assert isinstance(item["reason"], str) and len(item["reason"]) > 0
        
    surpluses = [it["tomorrow_expected_surplus_pct"] for it in items]
    print(f"\n[Challenge 3] Surplus progression across ranks:")
    for idx, (r, s, sym) in enumerate(zip(items, surpluses, [it["symbol"] for it in items])):
        print(f"  Rank {idx+1}: {sym:<8} surplus={s:+.2f}%")
        
    for i in range(len(surpluses) - 1):
        assert surpluses[i] > surpluses[i+1], (
            f"Ranking order is NOT strictly descending at rank {i+1} -> {i+2}: "
            f"{items[i]['symbol']} ({surpluses[i]}%) vs {items[i+1]['symbol']} ({surpluses[i+1]}%)"
        )


# ==============================================================================
# Challenge 4: Atomic Write Concurrency & Lock Contention on Windows
# ==============================================================================

def test_atomic_write_concurrency_and_race_condition(tmp_path: Path):
    """
    Adversarial Challenge 4:
    Simulate heavy concurrent read/write and unexpected interruptions.
    - Multiple threads simultaneously writing different versions to the same JSON file.
    - Multiple threads continuously reading and verifying JSON integrity.
    - Tests for Windows file locking PermissionError ([WinError 32] / [WinError 5]).
    """
    target_file = tmp_path / "concurrent_ranking.json"
    
    initial_payload = {"version": 0, "status": "INITIAL", "numbers": list(range(100))}
    atomic_write_json(str(target_file), initial_payload)
    
    stop_event = threading.Event()
    read_errors = []
    write_errors = []
    reads_completed = [0]
    writes_completed = [0]
    
    def writer_worker(thread_id: int):
        cnt = 0
        while not stop_event.is_set() and cnt < 30:
            cnt += 1
            payload = {
                "version": cnt,
                "writer_id": thread_id,
                "timestamp": time.time(),
                "data": [f"item_{k}_{thread_id}" for k in range(50)]
            }
            try:
                ok = atomic_write_json(str(target_file), payload)
                if not ok:
                    write_errors.append((thread_id, cnt, "atomic_write returned False"))
                else:
                    writes_completed[0] += 1
            except Exception as e:
                write_errors.append((thread_id, cnt, str(e)))
            time.sleep(0.005)
            
    def reader_worker(thread_id: int):
        while not stop_event.is_set():
            data = None
            try:
                for attempt in range(5):
                    try:
                        with open(str(target_file), "r", encoding="utf-8") as f:
                            content = f.read()
                        if not content.strip():
                            time.sleep(0.003)
                            continue
                        data = json.loads(content)
                        break
                    except (PermissionError, OSError):
                        time.sleep(0.003)
                    except json.JSONDecodeError:
                        time.sleep(0.003)

                if data is None:
                    # 若重试后仍未读到，最后尝试一次以捕获确切的读取异常
                    with open(str(target_file), "r", encoding="utf-8") as f:
                        content = f.read()
                    if not content.strip():
                        read_errors.append((thread_id, "Empty file read"))
                        continue
                    data = json.loads(content)

                if "version" not in data:
                    read_errors.append((thread_id, "Corrupted schema"))
                reads_completed[0] += 1
            except json.JSONDecodeError as jde:
                read_errors.append((thread_id, f"JSONDecodeError: {str(jde)}"))
            except Exception as e:
                read_errors.append((thread_id, f"ReadException: {str(e)}"))
            time.sleep(0.002)
            
    with ThreadPoolExecutor(max_workers=12) as executor:
        writer_futures = [executor.submit(writer_worker, wid) for wid in range(4)]
        reader_futures = [executor.submit(reader_worker, rid) for rid in range(8)]
        
        for f in writer_futures:
            f.result()
            
        stop_event.set()
        for f in reader_futures:
            f.result()
            
    print(f"\n[Challenge 4] Concurrency Results:")
    print(f"  Writes completed: {writes_completed[0]}")
    print(f"  Reads completed : {reads_completed[0]}")
    print(f"  Write errors    : {len(write_errors)}")
    print(f"  Read errors     : {len(read_errors)}")
    
    tmp_files = list(tmp_path.glob("*.tmp*"))
    print(f"  Leaked tmp files: {len(tmp_files)}")
    
    assert len(read_errors) == 0, (
        f"MEDIUM: Concurrent readers observed {len(read_errors)} corrupted/failed reads: {read_errors[:3]} "
        f"(Windows file lock collision during atomic replace without retry/sharing)."
    )
    assert len(write_errors) == 0, f"Concurrent writers encountered {len(write_errors)} errors: {write_errors[:3]}"
    assert len(tmp_files) == 0, f"Leaked tmp files remaining: {tmp_files}"


# ==============================================================================
# Challenge 5: Non-finite Price Inputs Corrupting JSON (RFC 8259 Violation)
# ==============================================================================

def test_nan_price_rfc8259_corruption():
    """
    Adversarial Challenge 5:
    When PolarStar or upstream data feed injects NaN (e.g. quote disconnected or halted),
    does compute_cross_asset_surplus sanitize it or serialize NaN into JSON?
    Standard JSON (RFC 8259) prohibits NaN / Infinity. Browser JSON.parse will crash.
    """
    res = compute_cross_asset_surplus(sa_price=float("nan"), fg_price=1100.0)
    
    sa_item = next((it for it in res if it["symbol"] == "SA701"), None)
    assert sa_item is not None
    
    print(f"\n[Challenge 5] SA701 last_price with NaN input: {sa_item['last_price']}, price string: '{sa_item['price']}'")
    
    # Check if last_price is NaN
    is_nan = math.isnan(float(sa_item["last_price"]))
    
    # If JSON is dumped, check if standard strict json parser can parse it
    raw_json = json.dumps(sa_item)
    
    # Standard JavaScript JSON does not allow NaN without quotes
    assert not is_nan, (
        f"LOW: Pipeline passed raw NaN into last_price. In JSON serialization: '{raw_json[:60]}...' "
        f"which creates non-standard RFC 8259 JSON and crashes frontend JSON.parse!"
    )


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
