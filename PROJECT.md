# Project: 郑商所纯碱/玻璃与A股产业链跨资产期望盈余 (TES) 系统重构

## Architecture
- **Layer 1: Causal Ontology (ADR-0002)**: Physical process node hierarchy (`ProcessNode`, `ValueAddedOntology`) spanning 4 physical tiers:
  - Node 1: Raw Material / Trona Mining (`NODE_01_TRONA` -> 000683 远兴能源)
  - Node 2: Intermediate Soda Ash (`NODE_02_SODA_SYN` / `NODE_03_SODA_SYN` -> 600328 中盐化工, 000822 山东海化, 600409 三友化工, SA0/SA701)
  - Node 3: Float & PV Glass Processing (`NODE_04_FLOAT_GLASS`, `NODE_05_PV_GLASS` -> 601636 旗滨集团, 600586 金晶科技, 601865 福莱特, 000012 南玻A, FG0/FG701)
  - Node 4: Terminal Strategic Demand (`NODE_07_END_MODULE`, `NODE_08_END_STORAGE` -> 601012 隆基绿能, 300750 宁德时代)
  - Chemical Stoichiometry: 1 ton FG consumes ~0.20 ton SA; ~350 RMB fuel cost; continuous float kiln rigidity.
- **Layer 2: Two-Layer Decoupling Engine (ADR-0001)**:
  - Qualitative Layer: LLM generates structured `QualitativeProposition` entities with evidence tags (`[FACT]`, `[OPINION]`, `[INFERENCE]`) and node coordinates. Regex enforces complete prohibition of numerical Alpha. Mandatory `[FACT]` tag gate rejects ungrounded rumors.
  - Quantitative Layer: Independent Python code gate executes Fama-MacBeth 2-stage regression with Newey-West HAC standard errors ($q = \max(1, \lfloor 4(T/100)^{2/9} \rfloor) \approx 7$).
  - One-Vote Veto Code Gate: Strict dual hurdle requiring $p < 0.05$ AND $IR \ge 0.30$. 100% veto on non-significant propositions. Complete division-by-zero protection ($\sigma(\varepsilon) \le 10^{-9} \implies IR = 0.0$).
- **Layer 3: Cross-Asset Expected Surplus (TES) Pipeline**:
  - Unifies commodity futures and equity surplus into standardized daily expected surplus percentages.
  - Dynamically couples synthetic soda ash margins to Trona cost shocks ensuring monotonicity.
  - Generates `02_极星量化与实盘/logs/ranking_cross_asset.json` with retry-protected atomic file write under Windows.
  - Adheres 100% to frontend contract (`docs/index.html`) across 10 table columns.

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | Causal Process Node Hierarchy | 4-tier process node definitions with stoichiometric relations ($k=0.20$) | M1 | ADR-0002 | DONE |
| 2 | Commodity & Equity Stock Mapping | Mapping SA, FG, and 12 core A-shares (000683, 600328, 601636, etc.) to nodes | M1 | ORIGINAL_REQUEST §R1 | DONE |
| 3 | Upstream/Downstream Query & Traversal | Graph traversal for cost shock propagation and margin expansion/compression | M1 | ADR-0002 / NALE | DONE |
| 4 | Structured Qualitative Proposition Schema | LLM proposition extraction data model (`QualitativeProposition`) with FOI tags & regex gate | M2 | ADR-0001 / ADR-0004 | DONE |
| 5 | Fama-MacBeth 2-Stage Regression | Time-series factor regression estimating asset Alpha, t-stat, and residuals | M2 | ADR-0001 | DONE |
| 6 | Newey-West Adaptive HAC Bandwidth | Bandwidth calculation $q = \max(1, \lfloor 4(T/100)^{2/9} \rfloor)$ for autocorrelation/heteroskedasticity | M2 | ADR-0001 / Report | DONE |
| 7 | Statistical Alpha Gate (One-Vote Veto) | Dual hurdle ($p < 0.05$ and $IR \ge 0.30$) with 100% veto on noise/spurious hype | M2 | ORIGINAL_REQUEST §R2 | DONE |
| 8 | Cross-Asset Expected Surplus Pipeline | Unified pipeline computing TES for commodity futures and industrial chain stocks | M3 | ORIGINAL_REQUEST §R3 | DONE |
| 9 | Frontend Ranking Contract & Atomic Write | Output `ranking_cross_asset.json` conforming to `docs/index.html` 10-column table | M3 | docs/index.html | DONE |
| 10 | Positive Causal Transmission Test Suite | Test Trona cost collapse & PV demand pull transmitting through value-added nodes | M4 | ORIGINAL_REQUEST §R3 | DONE |
| 11 | Counterfactual False Hype Interception Suite | Test 100% interception of rumors, sentiment hype, and noise failing code gate | M4 | ORIGINAL_REQUEST §R3 | DONE |
| 12 | End-to-End Test Suite & Zero Regression | 343 total tests pass with exit code 0 | M4 | Acceptance Criteria | DONE |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Causal Ontology of SA & FG Industry | `src/models/causal_ontology.py`, `tests/test_causal_ontology.py` | none | DONE |
| M2 | Two-Layer Decoupling Engine | `src/models/two_layer_engine.py`, `tests/test_two_layer_decoupling.py` | M1 interface | DONE |
| M3 | Cross-Asset TES Pipeline Integration | `02_极星量化与实盘/strategies/CZCE_19_5_DualProduct_Plugin.py`, `src/models/tes_pipeline.py` | M1, M2 | DONE |
| M4 | Automated Test Suite & E2E Validation | `tests/test_tes_e2e_pipeline.py`, full pytest suite regression (343 passed) | M1, M2, M3 | DONE |

## Interface Contracts
- Fully implemented and verified in `src/models/causal_ontology.py`, `src/models/two_layer_engine.py`, `src/models/tes_pipeline.py`, and `02_极星量化与实盘/strategies/CZCE_19_5_DualProduct_Plugin.py`.
- Output: `02_极星量化与实盘/logs/ranking_cross_asset.json` verified matching 10-column frontend contract in `docs/index.html`.
