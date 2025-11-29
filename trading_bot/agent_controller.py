import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from . import config, data, rl_agent, trade_manager

AgentAction = Literal[
    "HOLD",
    "OPEN_LONG",
    "OPEN_SHORT",
    "CLOSE_TRADE",
    "SCALE_IN",
    "SCALE_OUT",
    "HARD_STOP",
]


@dataclass
class AgentDecision:
    action: AgentAction
    symbol: str
    quantity: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trade_id: Optional[str] = None
    reason: str = ""
    metadata: Optional[Dict[str, Any]] = None


logger = logging.getLogger(__name__)


class MasterAgent:
    """
    Agente maestro que fusiona el modelo tradicional y la política RL.

    - Construye observaciones ricas con datos de mercado, cuenta y trades.
    - Delega en el RL para gestión de riesgo y tamaño mientras respeta
      guardarraíles duros de drawdown y límites por símbolo.
    - Devuelve acciones de alto nivel que el bot ejecuta de forma reactiva.
    """

    def __init__(self, rl: rl_agent.RLTradingAgent | None = None) -> None:
        self.enabled = bool(config.AGENT_CONTROL_ENABLED)
        self.rl = rl or rl_agent.RLTradingAgent()
        self._hard_stop = False

    # ---------- Observación del entorno ----------

    def build_observation(self, symbol: str, candidate_signal: dict | None = None) -> Dict[str, Any]:
        market = data.get_market_data(symbol, limit=50) or {}
        open_trades = [
            t for t in trade_manager.all_open_trades() if trade_manager.normalize_symbol(t.get("symbol", ""))
            == trade_manager.normalize_symbol(symbol)
        ]
        live_position = trade_manager.get_live_position(symbol)
        position_info = None
        if live_position:
            try:
                size = abs(float(live_position.get("contracts") or live_position.get("holdVolume") or 0.0))
            except (TypeError, ValueError):
                size = 0.0
            side_norm = self._normalize_side(live_position.get("side"))
            if size > 0 and side_norm:
                position_info = {
                    "size": size,
                    "side": side_norm,
                    "raw": live_position,
                }
        balance_snapshot = trade_manager.last_recorded_balance()
        account = {
            "free_usdt": balance_snapshot.get("free_usdt", 0.0),
            "timestamp": balance_snapshot.get("timestamp", 0.0),
            "daily_drawdown_pct": balance_snapshot.get("daily_drawdown_pct", 0.0),
        }

        observation = {
            "symbol": symbol,
            "market": market,
            "open_trades": open_trades,
            "live_position": position_info,
            "account": account,
            "candidate_signal": candidate_signal or {},
        }
        return observation

    def _vectorize_observation(self, obs: Dict[str, Any]) -> Any:
        try:
            return self.rl.encode_state(obs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Falling back to empty state for RL: %s", exc)
            return None

    # ---------- Decisión principal ----------

    def decide(self, symbol: str, candidate_signal: Dict[str, Any] | None = None) -> AgentDecision:
        if not self.enabled or self._hard_stop:
            return AgentDecision(action="HOLD", symbol=symbol, reason="agent_disabled")

        obs = self.build_observation(symbol, candidate_signal=candidate_signal)
        consistency_flag = self._check_consistency(obs)
        if consistency_flag:
            logger.warning("AGENT consistency issue for %s: %s", symbol, consistency_flag)
            return AgentDecision(action="HOLD", symbol=symbol, reason=consistency_flag)
        state_vec = self._vectorize_observation(obs)

        dir_info = self._direction_from_model(symbol, obs, candidate_signal)
        rl_decision = self.rl.decide_action(
            state_vec=state_vec,
            direction=dir_info["direction"],
            confidence=dir_info["confidence"],
            has_open_trades=bool(obs.get("open_trades")),
        )

        if self._violates_guardrails(symbol, obs):
            logger.warning("AGENT guardrails triggered for %s", symbol)
            if config.AGENT_ALLOW_HARD_STOP and config.AGENT_DISABLE_ON_MAX_DD:
                self._hard_stop = True
                return AgentDecision(action="HARD_STOP", symbol=symbol, reason="max_drawdown")
            return AgentDecision(action="HOLD", symbol=symbol, reason="risk_guardrails")

        return self._map_rl_to_decision(symbol, rl_decision, candidate_signal, obs)

    def _direction_from_model(self, symbol: str, obs: Dict[str, Any], candidate_signal: Dict[str, Any] | None) -> Dict[str, Any]:
        base_dir = "flat"
        confidence = 0.5
        if candidate_signal:
            side = candidate_signal.get("side", "").lower()
            base_dir = "long" if side.startswith("buy") else "short"
            prob = candidate_signal.get("prob_success")
            if prob is not None:
                confidence = float(prob)

        # Placeholder: if no modelo probabilístico, usamos la señal de estrategia
        direction = base_dir if confidence >= 0.5 else "flat"
        return {"direction": direction, "confidence": max(0.0, min(1.0, confidence))}

    def _check_consistency(self, obs: Dict[str, Any]) -> str | None:
        """Detect mismatches between local trades and live positions."""

        open_trades = obs.get("open_trades") or []
        live_position = obs.get("live_position")

        if live_position and not open_trades:
            return "position_without_trade"
        if open_trades and not live_position:
            return "trade_without_position"

        if open_trades and live_position:
            trade_side = self._normalize_side(open_trades[0].get("side"))
            pos_side = self._normalize_side(live_position.get("side"))
            if trade_side and pos_side and trade_side != pos_side:
                return "position_trade_side_mismatch"
        return None

    def _violates_guardrails(self, symbol: str, obs: Dict[str, Any]) -> bool:
        open_trades = obs.get("open_trades") or []
        live_position = obs.get("live_position")
        active_slots = max(len(open_trades), 1 if live_position else 0)
        if active_slots >= config.AGENT_MAX_TRADES_PER_SYMBOL:
            return True
        account = obs.get("account") or {}
        daily_dd_pct = float(account.get("daily_drawdown_pct") or 0.0)
        if daily_dd_pct <= -abs(config.AGENT_MAX_DAILY_LOSS_PCT):
            return True
        return False

    def _map_rl_to_decision(
        self,
        symbol: str,
        rl_decision: Dict[str, Any],
        candidate_signal: Dict[str, Any] | None,
        obs: Dict[str, Any],
    ) -> AgentDecision:
        action_type = rl_decision.get("action") or "HOLD"
        size_mult = float(rl_decision.get("size_mult", 1.0))
        tp_mult = float(rl_decision.get("tp_mult", 1.0))
        sl_mult = float(rl_decision.get("sl_mult", 1.0))
        metadata = {"rl_raw": rl_decision}

        if action_type == "HOLD":
            return AgentDecision(action="HOLD", symbol=symbol, reason="rl_hold", metadata=metadata)

        open_trades = obs.get("open_trades") or []
        live_position = obs.get("live_position")
        current_side = self._normalize_side(live_position.get("side")) if live_position else self._normalize_side(
            open_trades[0].get("side") if open_trades else ""
        )

        if action_type in ("OPEN_LONG", "OPEN_SHORT"):
            desired_side = "long" if action_type == "OPEN_LONG" else "short"
            if current_side:
                reason = "position_exists" if current_side == desired_side else "position_conflict"
                return AgentDecision(action="HOLD", symbol=symbol, reason=reason, metadata=metadata)
            base_qty = (candidate_signal or {}).get("quantity") or 0.0
            qty = max(base_qty * size_mult, 0.0)
            if qty <= 0:
                return AgentDecision(action="HOLD", symbol=symbol, reason="zero_qty", metadata=metadata)
            tp = (candidate_signal or {}).get("take_profit")
            sl = (candidate_signal or {}).get("stop_loss")
            if tp:
                tp *= tp_mult
            if sl:
                sl *= sl_mult
            return AgentDecision(
                action=action_type,
                symbol=symbol,
                quantity=qty,
                take_profit=tp,
                stop_loss=sl,
                reason="rl_open",
                metadata=metadata,
            )

        if action_type == "CLOSE_TRADE":
            if current_side:
                trade = sorted(open_trades, key=lambda t: t.get("created_ts", time.time()))[0] if open_trades else None
                return AgentDecision(
                    action="CLOSE_TRADE",
                    symbol=symbol,
                    trade_id=trade.get("trade_id") if trade else None,
                    reason="rl_close",
                    metadata=metadata,
                )
            return AgentDecision(action="HOLD", symbol=symbol, reason="no_live_position", metadata=metadata)

        if action_type == "SCALE_IN" and config.AGENT_ALLOW_SCALE_IN and current_side:
            trade = sorted(open_trades, key=lambda t: t.get("created_ts", time.time()))[0] if open_trades else None
            qty_source = trade if trade else live_position
            qty_base = float(qty_source.get("quantity") or qty_source.get("size") or 0.0) if qty_source else 0.0
            qty = qty_base * max(size_mult, 0.1)
            if qty <= 0:
                return AgentDecision(action="HOLD", symbol=symbol, reason="zero_qty_position", metadata=metadata)
            return AgentDecision(
                action="SCALE_IN",
                symbol=symbol,
                quantity=qty,
                trade_id=trade.get("trade_id") if trade else None,
                reason="rl_scale_in",
                metadata=metadata,
            )

        if action_type == "SCALE_OUT" and config.AGENT_ALLOW_SCALE_OUT and current_side:
            trade = sorted(open_trades, key=lambda t: t.get("created_ts", time.time()))[0] if open_trades else None
            qty_source = trade if trade else live_position
            base_qty = float(qty_source.get("quantity") or qty_source.get("size") or 0.0) if qty_source else 0.0
            qty = base_qty * max(min(size_mult, 0.9), 0.1)
            if qty <= 0:
                return AgentDecision(action="HOLD", symbol=symbol, reason="zero_qty_position", metadata=metadata)
            return AgentDecision(
                action="SCALE_OUT",
                symbol=symbol,
                quantity=qty,
                trade_id=trade.get("trade_id") if trade else None,
                reason="rl_scale_out",
                metadata=metadata,
            )

        return AgentDecision(action="HOLD", symbol=symbol, reason="fallback", metadata=metadata)

    def _normalize_side(self, side_value: Any) -> str:
        side = str(side_value or "").lower()
        if side.startswith("buy") or side == "long":
            return "long"
        if side.startswith("sell") or side == "short":
            return "short"
        return ""

    # ---------- Feedback tras cierre de trade ----------

    def on_trade_closed(self, trade: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.rl.record_trade_outcome(trade)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("AGENT RL record_trade_outcome error: %s", exc)
