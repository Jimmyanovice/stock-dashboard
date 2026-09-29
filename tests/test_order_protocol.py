# -*- coding: utf-8 -*-
"""文件握手协议与权限分离的结构性测试。

本测试套件**不需要极星客户端**即可运行，覆盖：
1. 订单意向的全部限额校验（手数/白名单/时效/审计字段）；
2. 人工批准凭证的字节级绑定、防重放与时效校验；
3. 策略文件的权限分离不变式（Stage-1 只读桥接不得出现任何下单函数）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.trading import order_protocol as proto

ROOT = Path(__file__).resolve().parent.parent
BRIDGE_PATH = ROOT / "02_极星量化与实盘" / "bridge" / "CZCE_AccountBridge.py"
GATEWAY_PATH = ROOT / "02_极星量化与实盘" / "strategies" / "CZCE_OrderGateway.py"
POLICY_PATH = ROOT / "02_极星量化与实盘" / "ipc" / "policy.json"


@pytest.fixture()
def ipc(tmp_path):
    """构造一个临时的 IPC 目录，policy.json 与仓库保持一致。"""
    (tmp_path / proto.POLICY_FILENAME).write_text(
        POLICY_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def policy(ipc):
    return proto.load_policy(ipc)


def _good_request(policy, **overrides):
    kwargs = dict(
        symbol="ZCE|Z|SA|MAIN",
        direction="SELL",
        offset="ENTRY",
        lots=1,
        reason="G3 通路验证：单笔 1 手测试单",
    )
    kwargs.update(overrides)
    return proto.build_request(policy, **kwargs)


# --------------------------------------------------------------------------
# 策略层
# --------------------------------------------------------------------------
class TestPolicy:
    def test_default_policy_version_matches_code(self, policy):
        assert policy["protocol_version"] == proto.PROTOCOL_VERSION

    def test_trading_disabled_by_default(self, policy):
        assert policy["allow_trading"] is False, "出厂策略必须默认禁止交易"

    def test_limits_are_conservative(self, policy):
        assert policy["max_lots_per_order"] <= 5
        assert policy["max_orders_per_day"] <= 10
        assert policy["max_open_orders"] <= 2
        assert policy["allowed_price_types"] == ["LIMIT"], "禁止市价单"

    def test_missing_policy_falls_back_to_defaults(self, tmp_path):
        p = proto.load_policy(tmp_path)
        assert p["allow_trading"] is False

    def test_version_mismatch_raises(self, tmp_path):
        (tmp_path / proto.POLICY_FILENAME).write_text(
            json.dumps({"protocol_version": "9.9"}), encoding="utf-8"
        )
        with pytest.raises(proto.ProtocolError):
            proto.load_policy(tmp_path)


# --------------------------------------------------------------------------
# 意向校验
# --------------------------------------------------------------------------
class TestRequestValidation:
    def test_valid_request_passes(self, policy):
        proto.validate_request(_good_request(policy), policy)

    def test_lots_over_cap_rejected(self, policy):
        req = _good_request(policy, lots=policy["max_lots_per_order"] + 1)
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(req, policy)
        assert any("上限" in r for r in exc.value.reasons)

    def test_zero_or_negative_lots_rejected(self, policy):
        for bad in (0, -1):
            with pytest.raises(proto.ProtocolError):
                proto.validate_request(_good_request(policy, lots=bad), policy)

    def test_symbol_outside_whitelist_rejected(self, policy):
        req = _good_request(policy, symbol="SHFE|F|CU|2512")
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(req, policy)
        assert any("白名单" in r for r in exc.value.reasons)

    def test_market_order_rejected(self, policy):
        req = _good_request(policy, price_type="MARKET")
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(req, policy)
        assert any("price_type" in r for r in exc.value.reasons)

    def test_expired_request_rejected(self, policy):
        req = _good_request(policy)
        req["created_at_epoch"] = time.time() - policy["max_request_age_sec"] - 1
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(req, policy)
        assert any("过期" in r for r in exc.value.reasons)

    def test_clock_skew_into_future_rejected(self, policy):
        req = _good_request(policy)
        req["created_at_epoch"] = time.time() + 3600
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(req, policy)
        assert any("未来" in r for r in exc.value.reasons)

    def test_empty_reason_rejected(self, policy):
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(_good_request(policy, reason="   "), policy)
        assert any("reason" in r for r in exc.value.reasons)

    def test_daily_order_cap_enforced(self, policy):
        req = _good_request(policy)
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(req, policy, orders_today=policy["max_orders_per_day"])
        assert any("当日委托数" in r for r in exc.value.reasons)

    def test_open_order_cap_enforced(self, policy):
        req = _good_request(policy)
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(req, policy, open_orders=policy["max_open_orders"])
        assert any("在途委托" in r for r in exc.value.reasons)

    def test_missing_request_rejected(self, policy):
        with pytest.raises(proto.ProtocolError):
            proto.validate_request(None, policy)

    def test_all_errors_reported_at_once(self, policy):
        bad = _good_request(policy, symbol="X", lots=99, reason="")
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_request(bad, policy)
        assert len(exc.value.reasons) >= 3, "应一次性回报全部问题而非只报第一个"


# --------------------------------------------------------------------------
# 人工批准凭证
# --------------------------------------------------------------------------
class TestApproval:
    def _prepare(self, ipc, policy):
        req = _good_request(policy)
        path = proto.write_json_atomic(ipc / proto.REQUEST_FILENAME, req)
        return req, proto.sha256_file(path)

    def test_valid_approval_passes(self, ipc, policy):
        req, sha = self._prepare(ipc, policy)
        appr = proto.make_approval(req, sha, "张三")
        proto.validate_approval(appr, req, sha, policy)

    def test_sha_mismatch_rejected(self, ipc, policy):
        req, sha = self._prepare(ipc, policy)
        appr = proto.make_approval(req, sha, "张三")
        appr["request_sha256"] = "0" * 64
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_approval(appr, req, sha, policy)
        assert any("篡改" in r for r in exc.value.reasons)

    def test_request_id_mismatch_rejected(self, ipc, policy):
        req, sha = self._prepare(ipc, policy)
        appr = proto.make_approval(req, sha, "张三")
        appr["request_id"] = "req-20260101T000000Z-abcdef"
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_approval(appr, req, sha, policy)
        assert any("不一致" in r for r in exc.value.reasons)

    def test_tampered_request_invalidates_approval(self, ipc, policy):
        """批准后再改动意向文件，哈希必须失配。"""
        req, sha = self._prepare(ipc, policy)
        appr = proto.make_approval(req, sha, "张三")

        req["lots"] = 1
        req["reason"] = "被篡改的理由"
        proto.write_json_atomic(ipc / proto.REQUEST_FILENAME, req)
        new_sha = proto.sha256_file(ipc / proto.REQUEST_FILENAME)

        assert new_sha != sha
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_approval(appr, req, new_sha, policy)
        assert any("篡改" in r for r in exc.value.reasons)

    def test_weak_token_rejected(self, ipc, policy):
        req, sha = self._prepare(ipc, policy)
        appr = proto.make_approval(req, sha, "张三")
        appr["token"] = "abc"
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_approval(appr, req, sha, policy)
        assert any("令牌" in r for r in exc.value.reasons)

    def test_missing_approver_rejected(self, ipc, policy):
        req, sha = self._prepare(ipc, policy)
        appr = proto.make_approval(req, sha, "张三")
        appr["approved_by"] = "  "
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_approval(appr, req, sha, policy)
        assert any("approved_by" in r for r in exc.value.reasons)

    def test_stale_approval_rejected(self, ipc, policy):
        req, sha = self._prepare(ipc, policy)
        appr = proto.make_approval(req, sha, "张三")
        appr["approved_at_epoch"] = time.time() - policy["max_request_age_sec"] - 1
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_approval(appr, req, sha, policy)
        assert any("过期" in r for r in exc.value.reasons)

    def test_missing_approval_is_not_silently_allowed(self, ipc, policy):
        req, sha = self._prepare(ipc, policy)
        with pytest.raises(proto.ProtocolError) as exc:
            proto.validate_approval(None, req, sha, policy)
        assert any("不存在" in r for r in exc.value.reasons)

    def test_token_is_unique_per_approval(self, ipc, policy):
        req, sha = self._prepare(ipc, policy)
        tokens = {proto.make_approval(req, sha, "张三")["token"] for _ in range(50)}
        assert len(tokens) == 50, "批准令牌必须每次唯一，否则可被重放"


# --------------------------------------------------------------------------
# 熔断与文件写入
# --------------------------------------------------------------------------
class TestKillSwitchAndIO:
    def test_kill_switch_detected(self, ipc):
        assert proto.kill_switch_active(ipc) is False
        (ipc / proto.KILL_SWITCH_FILENAME).write_text("stop", encoding="utf-8")
        assert proto.kill_switch_active(ipc) is True

    def test_atomic_write_roundtrip(self, ipc):
        payload = {"a": 1, "中文": "值"}
        target = ipc / "sample.json"
        proto.write_json_atomic(target, payload)
        assert proto.read_json(target) == payload
        assert not (ipc / "sample.json.tmp").exists(), "临时文件必须被清理"

    def test_corrupt_json_raises(self, ipc):
        (ipc / "broken.json").write_bytes(b"{not json")
        with pytest.raises(proto.ProtocolError):
            proto.read_json(ipc / "broken.json")

    def test_empty_json_raises(self, ipc):
        (ipc / "empty.json").write_bytes(b"   ")
        with pytest.raises(proto.ProtocolError):
            proto.read_json(ipc / "empty.json")

    def test_missing_file_returns_none(self, ipc):
        assert proto.read_json(ipc / "nope.json") is None


# --------------------------------------------------------------------------
# 结构性不变式：权限分离
# --------------------------------------------------------------------------
class TestPrivilegeSeparation:
    """Stage-1 桥接必须零发单能力；Stage-2 网关必须受 policy 约束。

    使用 AST 而非文本匹配：注释与文档字符串中提到函数名不应导致误报。
    """

    @staticmethod
    def _called_names(path):
        """收集被调用或被当作函数对象传递的极星 API 名称。

        策略里统一用 `_safe(A_SendOrder, ...)` 的容错包装，因此 API 名会作为
        **实参**出现而不是调用目标，必须一并收集。
        """
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    names.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    names.add(fn.attr)
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        names.add(arg.id)
        return names

    def test_bridge_contains_no_order_apis(self):
        called = self._called_names(BRIDGE_PATH)
        forbidden = {
            "A_SendOrder", "A_DeleteOrder", "A_ModifyOrder", "A_SendOrderBatch",
            "SellShort", "BuyToCover", "Sell", "Buy", "StartTrade", "DeleteAllOrders",
        }
        overlap = called & forbidden
        assert not overlap, "只读桥接实际调用了下单类 API: %s" % sorted(overlap)

    def test_bridge_writes_only_readonly_artifacts(self):
        called = self._called_names(BRIDGE_PATH)
        assert "_write_json" in called
        src = BRIDGE_PATH.read_text(encoding="utf-8")
        assert "order_request.json" not in src
        assert "order_approval.json" not in src
        assert "cancel_request.json" not in src

    def test_gateway_uses_official_send_api(self):
        called = self._called_names(GATEWAY_PATH)
        assert "A_SendOrder" in called
        assert "A_DeleteOrder" in called

    def test_gateway_never_creates_its_own_approval(self):
        """网关只能读取批准凭证，绝不能自行写入，否则人工确认形同虚设。"""
        import ast

        tree = ast.parse(GATEWAY_PATH.read_text(encoding="utf-8"))
        writers = {"_write_json", "write_json_atomic", "_write_json_atomic"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args:
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                target = node.args[0]
                if name in writers and isinstance(target, ast.Name):
                    assert target.id != "APPROVAL_PATH", (
                        "网关不得写入 order_approval.json"
                    )

    def test_gateway_honours_kill_switch_and_live_toggle(self):
        called = self._called_names(GATEWAY_PATH)
        assert "StopTrade" in called, "熔断时必须调用 StopTrade"
        assert "StartTrade" in called, "恢复时必须重新开启实盘发单"
        src = GATEWAY_PATH.read_text(encoding="utf-8")
        assert "KILL_SWITCH_PATH" in src

    def test_gateway_reads_limits_from_policy_not_hardcoded(self):
        src = GATEWAY_PATH.read_text(encoding="utf-8")
        for key in ("max_lots_per_order", "max_orders_per_day", "max_open_orders",
                    "max_request_age_sec", "allowed_symbols", "min_available_for_entry"):
            assert key in src, "限额 %s 必须读自 policy.json" % key

    def test_both_strategies_are_python37_compatible(self):
        """极星内嵌的是 Python 3.7：禁用海象运算符与仅位置参数。"""
        import ast

        for path in (BRIDGE_PATH, GATEWAY_PATH):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                assert not isinstance(node, ast.NamedExpr), (
                    "%s 使用了海象运算符（需 Python 3.8+）" % path.name
                )
                if isinstance(node, ast.arguments):
                    assert not getattr(node, "posonlyargs", []), (
                        "%s 使用了仅位置参数（需 Python 3.8+）" % path.name
                    )


# --------------------------------------------------------------------------
# 人类可读摘要
# --------------------------------------------------------------------------
class TestDescribe:
    def test_summary_contains_key_fields(self, policy):
        req = _good_request(policy)
        text = proto.describe_request(req)
        assert req["request_id"] in text
        assert "卖出" in text and "开仓" in text
        assert "1 手" in text

    def test_empty_request_summary(self):
        assert "无订单意向" in proto.describe_request({})
