import argparse
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(env_path)

import logging
import os
import time
from collections import deque
from statistics import mean
from threading import Thread, RLock
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import (
    config,
    data,
    execution,
    strategy,
    webapp,
    notify,
    history,
    optimizer,
    permissions,
    trade_manager,
    shadow,
    shutdown,
    mode as bot_mode,

)
from .trade_manager import (
    add_trade,
    close_trade,
    find_trade,
    update_trade,
    all_open_trades,
    load_trades,
    save_trades,
    count_open_trades,
    count_trades_for_symbol,
    set_trade_state,
)
from .state_machine import TradeState
from .metrics import (
    start_metrics_server,
    update_trade_metrics,
    record_model_performance,
    maybe_alert,
)
from .monitor import monitor_system
from .utils import normalize_symbol

if not config.BITGET_API_KEY:
    print("\u26a0\ufe0f  No se carg\xf3 la API KEY. Revisa si el archivo .env existe o el gestor de secretos est\xe1 configurado.")
else:
    print("\u2705 Archivo .env cargado correctamente.")

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """Track predictive performance and adjust model weight when degraded."""

    def __init__(
        self,
        window: int,
        min_samples: int,
        min_win_rate: float,
        max_drift: float,
    ) -> None:
        self.samples: deque[tuple[float, int]] = deque(maxlen=window)
        self.min_samples = min_samples
        self.min_win_rate = min_win_rate
        self.max_drift = max_drift
        self._lock = RLock()
        self._degraded = False

    def reset(self) -> None:
        with self._lock:
            self.samples.clear()
            self._degraded = False

    def record_trade(self, trade: dict | None) -> None:
        if not trade:
            return
        prob = trade.get("model_prob")
        if prob is None:
            return
        outcome = 1 if float(trade.get("profit", 0.0)) > 0 else 0
        with self._lock:
            self.samples.append((float(prob), outcome))
            self._evaluate_locked()

    def _evaluate_locked(self) -> None:
        if len(self.samples) < max(self.min_samples, 1):
            return
        avg_prob = mean(p for p, _ in self.samples)
        win_rate = mean(outcome for _, outcome in self.samples)
        drift = abs(avg_prob - win_rate)
        record_model_performance(win_rate, avg_prob, drift)

        current_weight = strategy.get_model_weight()
        if (
            (win_rate < self.min_win_rate or drift > self.max_drift)
            and current_weight > config.MODEL_WEIGHT_FLOOR
        ):
            new_weight = max(
                config.MODEL_WEIGHT * config.MODEL_WEIGHT_DEGRADATION,
                config.MODEL_WEIGHT_FLOOR,
            )
            strategy.set_model_weight_override(new_weight)
            self._degraded = True
            logger.warning(
                "Model weight degraded to %.2f due to drift (win_rate=%.2f avg_prob=%.2f drift=%.2f)",
                new_weight,
                win_rate,
                avg_prob,
                drift,
            )
            self.samples.clear()
            return

        if self._degraded and win_rate >= self.min_win_rate and drift <= self.max_drift:
            strategy.set_model_weight_override(None)
            self._degraded = False
            logger.info(
                "Model performance recovered (win_rate=%.2f avg_prob=%.2f); restoring weight %.2f",
                win_rate,
                avg_prob,
                config.MODEL_WEIGHT,
            )
            self.samples.clear()


@dataclass
class ShadowPosition:
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    take_profit: float
    stop_loss: float
    open_time: datetime
    max_duration: timedelta
    probabilities: dict[str, float]


SHADOW_COMPARE_MODES = ("heuristic", "hybrid")
_shadow_positions: dict[str, ShadowPosition] = {}
_shadow_lock = RLock()


_model_lock = RLock()
_cached_model = None
_cached_model_mtime: float | None = None
_model_missing_logged = False


def _set_cached_model(model) -> None:
    global _cached_model
    with _model_lock:
        _cached_model = model


def _model_manifest_timestamp(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def maybe_reload_model(force: bool = False) -> None:
    """Reload the predictive model if the file changed."""

    global _cached_model_mtime, _model_missing_logged
    model_path = config.MODEL_PATH
    mtime = _model_manifest_timestamp(model_path)
    if mtime is None:
        if force or (_cached_model is not None):
            logger.warning("Modelo no disponible en %s", model_path)
        _set_cached_model(None)
        _cached_model_mtime = None
        _model_missing_logged = True
        return

    if not force and _cached_model_mtime is not None and mtime <= _cached_model_mtime:
        return

    model = optimizer.load_model(model_path)
    if model is None:
        if not _model_missing_logged:
            logger.warning(
                "No se encontró el modelo histórico en %s; las señales se generarán sin ajuste.",
                model_path,
            )
            _model_missing_logged = True
        _set_cached_model(None)
        _cached_model_mtime = None
        return

    _set_cached_model(model)
    _cached_model_mtime = mtime
    _model_missing_logged = False
    strategy.set_model_weight_override(None)
    MODEL_MONITOR.reset()
    logger.info("Modelo predictivo recargado desde %s", model_path)


def current_model():
    with _model_lock:
        return _cached_model


MODEL_MONITOR = ModelPerformanceMonitor(
    config.MODEL_PERFORMANCE_WINDOW,
    config.MODEL_MIN_SAMPLES_FOR_MONITOR,
    config.MODEL_MIN_WIN_RATE,
    config.MODEL_MAX_CALIBRATION_DRIFT,
)


def _reset_shadow_positions() -> None:
    with _shadow_lock:
        _shadow_positions.clear()


def _record_shadow_payloads(position: ShadowPosition, signal: dict) -> None:
    base_payload = dict(signal)
    for mode in SHADOW_COMPARE_MODES:
        payload = dict(base_payload)
        if mode == "heuristic":
            prob = position.probabilities.get(mode)
            if "orig_prob" in payload:
                payload["prob_success"] = payload["orig_prob"]
            elif prob is not None:
                payload["prob_success"] = prob
        else:
            prob = position.probabilities.get(mode)
            if prob is not None:
                payload["prob_success"] = prob
        payload["trade_id"] = f"{position.trade_id}:{mode}"
        payload["mode"] = mode
        shadow.record_shadow_signal(position.symbol, mode, payload)


def _register_shadow_trade(signal: dict) -> None:
    trade_uuid = str(uuid.uuid4())
    entry_price = float(signal.get("entry_price", 0.0))
    quantity = float(signal.get("quantity", 0.0))
    take_profit = float(signal.get("take_profit", entry_price))
    stop_loss = float(signal.get("stop_loss", entry_price))
    try:
        max_minutes = int(
            signal.get("max_duration_minutes", config.MAX_TRADE_DURATION_MINUTES)
        )
    except (TypeError, ValueError):
        max_minutes = config.MAX_TRADE_DURATION_MINUTES
    if max_minutes <= 0:
        max_minutes = config.MAX_TRADE_DURATION_MINUTES

    probabilities = {
        "hybrid": float(signal.get("prob_success", 0.0)),
        "heuristic": float(
            signal.get("orig_prob", signal.get("prob_success", 0.0))
        ),
    }

    position = ShadowPosition(
        trade_id=trade_uuid,
        symbol=str(signal.get("symbol", "")),
        side=str(signal.get("side", "BUY")),
        entry_price=entry_price,
        quantity=quantity,
        take_profit=take_profit,
        stop_loss=stop_loss,
        open_time=datetime.now(timezone.utc),
        max_duration=timedelta(minutes=max_minutes),
        probabilities=probabilities,
    )

    with _shadow_lock:
        _shadow_positions[trade_uuid] = position

    _record_shadow_payloads(position, signal)


def _assess_shadow_close(
    position: ShadowPosition, price: float, now: datetime
) -> tuple[str | None, float, float]:
    side = position.side.upper()
    hit_tp = price >= position.take_profit if side == "BUY" else price <= position.take_profit
    hit_sl = price <= position.stop_loss if side == "BUY" else price >= position.stop_loss

    reason: str | None = None
    if hit_tp:
        reason = "TP"
    elif hit_sl:
        reason = "SL"
    elif now - position.open_time >= position.max_duration:
        reason = "MAX_DURATION"

    if reason is None:
        return None, price, 0.0

    if side == "BUY":
        profit = (price - position.entry_price) * position.quantity
    else:
        profit = (position.entry_price - price) * position.quantity

    return reason, price, profit


def _finalize_shadow_trade(
    position: ShadowPosition,
    reason: str,
    exit_price: float,
    profit: float,
    closed_at: datetime,
) -> None:
    payload = {
        "symbol": position.symbol,
        "side": position.side,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "quantity": position.quantity,
        "profit": profit,
        "close_reason": reason,
        "closed_at": closed_at.isoformat().replace("+00:00", "Z"),
    }
    for mode in SHADOW_COMPARE_MODES:
        result = dict(payload)
        prob = position.probabilities.get(mode)
        if prob is not None:
            result["prob_success"] = prob
        result["mode"] = mode
        shadow.finalize_shadow_trade(f"{position.trade_id}:{mode}", result)


def _process_shadow_positions() -> float:
    with _shadow_lock:
        snapshot = list(_shadow_positions.items())

    realized = 0.0
    completed: list[str] = []
    for trade_id, position in snapshot:
        price_raw = data.get_current_price_ticker(position.symbol)
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            continue

        now = datetime.now(timezone.utc)
        reason, exit_price, profit = _assess_shadow_close(position, price, now)
        if reason is None:
            continue

        _finalize_shadow_trade(position, reason, exit_price, profit, now)
        logger.info(
            "Shadow trade %s %s closed (%s) profit %.4f",
            position.symbol,
            position.side,
            reason,
            profit,
        )
        realized += profit
        completed.append(trade_id)

    if completed:
        with _shadow_lock:
            for trade_id in completed:
                _shadow_positions.pop(trade_id, None)

    return realized


def parse_args():
    """Parse CLI options controlling runtime mode selection."""

    parser = argparse.ArgumentParser(description="Trading bot runner")
    parser.add_argument(
        "--mode",
        choices=sorted(bot_mode.MODES.keys()),
        help="Selecciona modo de ejecución (override de ENV/menú)",
    )
    parser.add_argument(
        "--no-interactive-menu",
        action="store_true",
        help="Desactiva el menú interactivo incluso si hay TTY",
    )
    parser.add_argument(
        "--backtest-config",
        type=str,
        help="Ruta al YAML de configuración del backtest (modo backtest)",
    )
    parser.add_argument(
        "--backtest-data",
        type=str,
        help="Ruta al CSV con datos para el backtest (modo backtest)",
    )
    return parser.parse_args()


def _run_backtest_startup(args) -> None:
    """Execute the backtest runner when the selected mode requests it."""

    from . import backtest

    cfg_path_str = (
        args.backtest_config
        or config.BACKTEST_CONFIG_PATH
        or os.getenv("BACKTEST_CONFIG_PATH")
        or "backtest.yml"
    )
    data_path_str = (
        args.backtest_data
        or config.BACKTEST_DATA_PATH
        or os.getenv("BACKTEST_DATA_PATH")
        or "backtest.csv"
    )
    cfg_path = Path(cfg_path_str)
    data_path = Path(data_path_str)

    if not cfg_path.exists():
        logger.error("Backtest config not found: %s", cfg_path)
        return
    if not data_path.exists():
        logger.error("Backtest dataset not found: %s", data_path)
        return

    try:
        cfg = backtest._load_config(cfg_path)
        dataset = backtest._load_dataset(data_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unable to load backtest resources: %s", exc)
        return

    kpis = backtest.run_backtest(cfg, dataset)
    logger.info("Backtest KPIs: %s", kpis)


def open_new_trade(signal: dict):
    """Open a position and track its state via ``trade_manager``."""
    symbol = signal["symbol"]
    raw = normalize_symbol(symbol).replace("_", "")

    if config.SHADOW_MODE:
        _register_shadow_trade(signal)
        return None

    # Register trade in pending state first
    trade = add_trade(signal)
    try:
        execution.setup_leverage(execution.exchange, raw, signal["leverage"])
        order = execution.open_position(
            symbol,
            signal["side"],
            signal["quantity"],
            signal["entry_price"],
            order_type="limit",
        )
        if not isinstance(order, dict):
            logger.warning("Order response unexpected for %s: %s", symbol, order)
            set_trade_state(trade["trade_id"], TradeState.FAILED)
            return None

        order_id = order.get("id")
        avg_price = float(order.get("average") or signal["entry_price"])
        update_trade(
            trade["trade_id"],
            order_id=order_id,
            entry_price=avg_price,
            status="active",
        )

        status = execution.fetch_order_status(order_id, symbol)
        if status == "filled":
            set_trade_state(trade["trade_id"], TradeState.OPEN)
        elif status == "partial":
            # promote to OPEN first, then PARTIALLY_FILLED
            set_trade_state(trade["trade_id"], TradeState.OPEN)
            set_trade_state(trade["trade_id"], TradeState.PARTIALLY_FILLED)
        # else: remain in PENDING

        if status in {"filled", "partial"}:
            details = execution.get_order_fill_details(order_id, symbol)
            if details:
                updates: dict[str, float] = {}
                filled_qty = details.get("filled")
                if filled_qty is not None and filled_qty > 0:
                    updates["quantity"] = filled_qty
                remaining_qty = details.get("remaining")
                if remaining_qty is not None:
                    updates["remaining_quantity"] = max(remaining_qty, 0.0)
                avg_exec = details.get("average")
                if avg_exec:
                    updates.setdefault("entry_price", avg_exec)
                if updates:
                    update_trade(trade["trade_id"], **updates)

        save_trades()
        return find_trade(trade_id=trade["trade_id"])

    except permissions.PermissionError as exc:
        logger.error("Permission denied opening %s: %s", symbol, exc)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    except execution.OrderSubmitError:
        logger.error("Order submission failed for %s", symbol)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    except Exception as exc:
        logger.error("Error processing %s: %s", symbol, exc)
        set_trade_state(trade["trade_id"], TradeState.FAILED)
    return None


def close_existing_trade(
    trade: dict, reason: str
) -> tuple[dict | None, float | None, float | None]:
    """Close a trade enforcing state transitions and return results."""

    trade_id = trade.get("trade_id")
    symbol = trade.get("symbol")
    qty = float(trade.get("quantity", 0.0))
    close_side = "close_short" if trade.get("side") == "SELL" else "close_long"

    set_trade_state(trade_id, TradeState.CLOSING)
    try:
        order = execution.close_position(symbol, close_side, qty, order_type="market")
    except execution.OrderSubmitError:
        logger.error("Order submission failed when closing %s", symbol)
        set_trade_state(trade_id, TradeState.FAILED)
        save_trades()
        return None, None, None


    order_id = order.get("id")
    exec_price = float(order.get("average") or order.get("price") or trade.get("entry_price", 0.0))

    if order_id:
        attempts = max(1, config.API_RETRY_ATTEMPTS)
        for attempt in range(attempts):
            status = execution.fetch_order_status(order_id, symbol)
            if status == "filled":
                break
            if status == "partial":
                details = execution.get_order_fill_details(order_id, symbol)
                remaining_qty = 0.0
                if details:
                    remaining_qty = max(details.get("remaining", 0.0) or 0.0, 0.0)
                    if details.get("average"):
                        exec_price = float(details["average"])
                if remaining_qty <= config.CLOSE_REMAINING_TOLERANCE:
                    break
                logger.warning(
                    "Partial close for %s (remaining %.6f); reattempting", symbol, remaining_qty
                )
                try:
                    order = execution.close_position(
                        symbol, close_side, remaining_qty, order_type="market"
                    )
                    order_id = order.get("id", order_id)
                    if order.get("average"):
                        exec_price = float(order.get("average"))
                except execution.OrderSubmitError:
                    logger.error("Failed to close remaining %.6f for %s", remaining_qty, symbol)
                    break
            else:
                break
            time.sleep(0.5 * (attempt + 1))

    remaining = execution.fetch_position_size(symbol)
    if remaining > config.CLOSE_REMAINING_TOLERANCE:
        logger.warning(
            "Remanente %.6f detectado tras cierre de %s; trade marcado como parcial",
            remaining,
            symbol,
        )
        update_trade(trade_id, quantity=remaining)
        set_trade_state(trade_id, TradeState.PARTIALLY_FILLED)
        save_trades()
        return None, exec_price, None

    entry_price = float(trade.get("entry_price", exec_price))
    if trade.get("side") == "BUY":
        realized = (exec_price - entry_price) * qty
    else:
        realized = (entry_price - exec_price) * qty

    closed = close_trade(
        trade_id=trade_id,
        reason=reason,
        exit_price=exec_price,
        profit=realized,
    )
    MODEL_MONITOR.record_trade(closed)
    save_trades()
    return closed, exec_price, realized


def run_one_iteration_open(model=None):
    """Execute a single iteration of the opening logic."""
    if model is None:
        model = current_model()
    execution.cleanup_old_orders()
    if count_open_trades() >= config.MAX_OPEN_TRADES:
        return
    symbols = data.get_common_top_symbols(execution.exchange, 15)
    candidates = []
    seen = set()
    for symbol in symbols:
        if find_trade(symbol=symbol) or trade_manager.in_cooldown(symbol):
            continue
        norm = trade_manager.normalize_symbol(symbol)
        if norm in seen:
            continue
        seen.add(norm)
        raw = symbol.replace("_", "")
        if raw in config.BLACKLIST_SYMBOLS or raw in config.UNSUPPORTED_SYMBOLS:
            continue
        sig = strategy.decidir_entrada(symbol, modelo_historico=model)
        if not sig:
            continue
        if sig.get("risk_reward", 0) < config.MIN_RISK_REWARD:
            logger.debug(
                "Skip %s: RR=%.2f < %.2f", symbol, sig.get("risk_reward", 0), config.MIN_RISK_REWARD
            )
            continue
        if sig.get("quantity", 0) < config.MIN_POSITION_SIZE:
            logger.debug(
                "Skip %s: qty=%.8f < min=%.8f",
                symbol,
                sig.get("quantity", 0),
                config.MIN_POSITION_SIZE,
            )
            continue
        candidates.append(sig)

    candidates.sort(key=lambda s: s.get("prob_success", 0), reverse=True)
    for sig in candidates:
        if count_open_trades() >= config.MAX_OPEN_TRADES:
            break
        try:
            open_new_trade(sig)
        except Exception as exc:
            logger.error("Error processing %s: %s", sig.get("symbol"), exc)

def run():
    load_trades()  # Restaurar operaciones guardadas
    permissions.audit_environment(execution.exchange)

    # Sincronizar con posiciones reales en el exchange
    positions = execution.fetch_positions()
    active_symbols = set()
    for pos in positions:
        symbol = normalize_symbol(pos.get("symbol", ""))
        qty = float(pos.get("contracts", 0))
        if qty == 0:
            continue
        active_symbols.add(symbol)
        trade = find_trade(symbol=symbol)
        if trade:
            trade.update(
                quantity=qty,
                entry_price=float(pos.get("entryPrice", trade.get("entry_price", 0))),
                leverage=int(pos.get("leverage", trade.get("leverage", 1))),
                status="active",
            )
        else:
            trade = {
                "symbol": symbol,
                "side": "BUY" if pos.get("side") == "long" else "SELL",
                "quantity": qty,
                "entry_price": float(pos.get("entryPrice", 0)),
                "stop_loss": float(pos.get("stopLossPrice", 0)) or float(pos.get("entryPrice", 0)) * 0.98,
                "take_profit": float(pos.get("takeProfitPrice", 0)) or float(pos.get("entryPrice", 0)) * 1.02,
                "leverage": int(pos.get("leverage", 1)),
                "status": "active",
            }
            add_trade(trade)

    # Eliminar operaciones locales que no existan en el exchange
    for tr in list(all_open_trades()):
        if tr["symbol"] not in active_symbols:
            close_trade(trade_id=tr.get("trade_id"), reason="sync")

    # Cancelar órdenes abiertas pendientes no registradas
    for order in execution.fetch_open_orders():
        sym = normalize_symbol(order.get("symbol", ""))
        if sym not in active_symbols:
            execution.cancel_order(order.get("id"), sym)

    # Remove any stale pending orders
    execution.cleanup_old_orders()

    # Persist state after initial synchronization
    save_trades()

    # Launch the dashboard using trade_manager as the single source of trades
    # (no operations list is passed).
    Thread(
        target=webapp.start_dashboard,
        args=(config.WEBAPP_HOST, config.WEBAPP_PORT),
        daemon=True,
    ).start()

    # Expose Prometheus metrics and start system monitor
    Thread(target=start_metrics_server, args=(8001,), daemon=True).start()
    Thread(target=monitor_system, daemon=True).start()

    strategy.start_liquidity()

    shutdown.install_signal_handlers()
    shutdown.register_callback(save_trades)
    shutdown.register_callback(execution.cancel_all_orders)

    maybe_reload_model(force=True)
    daily_profit = 0.0
    trading_active = bool(config.ENABLE_TRADING or config.SHADOW_MODE)

    current_day = datetime.now(timezone.utc).date()
    loss_limit = -abs(config.DAILY_RISK_LIMIT)
    standby_notified = False
    shutdown_handled = False

    # Initial metric update
    update_trade_metrics(count_open_trades(), len(trade_manager.all_closed_trades()))

    logger.info("Starting trading loop...")

    while True:
        try:

            execution.cleanup_old_orders()

            maybe_reload_model()
            model = current_model()

            if shutdown.shutdown_requested() and not shutdown_handled:
                logger.info("Shutdown requested; stopping trading loop")
                trading_active = False
                shutdown_handled = True
                shutdown.execute_callbacks()
                break

            now = datetime.now(timezone.utc)
            if now.date() != current_day:
                current_day = now.date()
                daily_profit = 0.0
                trading_active = bool(config.ENABLE_TRADING or config.SHADOW_MODE)

                standby_notified = False
                logger.info("New trading day detected; counters reset")

            # Detener nuevas entradas cuando la pérdida diaria alcanza el límite
            # Se acepta que DAILY_RISK_LIMIT pueda definirse como valor negativo
            # (por ejemplo -50.0) o positivo (50.0). Siempre se compara contra
            # el valor negativo correspondiente.
            if daily_profit <= loss_limit and trading_active:
                trading_active = False
                standby_notified = False
                logger.error("Daily loss limit reached %.2f", daily_profit)
                maybe_alert(True, f"Daily loss limit reached {daily_profit:.2f} USDT")


            # ABRIR NUEVAS OPERACIONES (solo si hay hueco y permitido)
            allow_candidates = (
                trading_active
                and not config.MAINTENANCE
                and (config.ENABLE_TRADING or config.SHADOW_MODE)
                and count_open_trades() < config.MAX_OPEN_TRADES
            )

            if allow_candidates:
                if not config.SHADOW_MODE and not permissions.can_open_trade(execution.exchange):
                    logger.debug("Skipping entries: live trading permissions not granted")
                else:
                    if config.TEST_MODE and config.TEST_SYMBOLS:
                        symbols = [s.replace("/", "_").replace("-", "_")
                                   for s in config.TEST_SYMBOLS]
                    else:
                        symbols = data.get_common_top_symbols(execution.exchange, 15)
                    candidates = []
                    seen = set()
                    for symbol in symbols:
                        # enforce per-symbol trade limit
                        if count_trades_for_symbol(symbol) >= config.MAX_TRADES_PER_SYMBOL:
                            continue
                        raw = normalize_symbol(symbol).replace("_", "")
                        bl = {normalize_symbol(s).replace("_", "") for s in config.BLACKLIST_SYMBOLS}
                        unsup = {normalize_symbol(s).replace("_", "") for s in config.UNSUPPORTED_SYMBOLS}
                        if raw in bl or raw in unsup:
                            continue
                        if trade_manager.in_cooldown(symbol):
                            logger.info("Cooldown activo para %s; se omite nueva entrada", symbol)
                            continue
                        sig = strategy.decidir_entrada(symbol, modelo_historico=model)
                        if not sig:
                            continue
                        if sig.get("risk_reward", 0) < config.MIN_RISK_REWARD:
                            logger.debug(
                                "Skip %s: RR=%.2f < %.2f",
                                symbol,
                                sig.get("risk_reward", 0),
                                config.MIN_RISK_REWARD,
                            )
                            continue
                        if sig.get("quantity", 0) < config.MIN_POSITION_SIZE:
                            logger.debug(
                                "Skip %s: qty=%.8f < min=%.8f",
                                symbol,
                                sig.get("quantity", 0),
                                config.MIN_POSITION_SIZE,
                            )
                            continue
                        candidates.append(sig)

                    candidates.sort(key=lambda s: s.get("prob_success", 0), reverse=True)
                    for sig in candidates:
                        if count_open_trades() >= config.MAX_OPEN_TRADES:
                            break
                        symbol = sig["symbol"]
                        raw = normalize_symbol(symbol).replace("_", "")
                        try:
                            trade = open_new_trade(sig)
                            if not trade:
                                continue
                            notify.send_telegram(
                                f"Opened {symbol} {trade['side']} @ {trade['entry_price']}"
                            )
                            notify.send_discord(
                                f"Opened {symbol} {trade['side']} @ {trade['entry_price']}"
                            )
                            logger.info("Opened trade: %s", trade)
                        except permissions.PermissionError as exc:
                            logger.error("Permission denied for %s: %s", symbol, exc)
                            break
                        except Exception as exc:
                            logger.error("Error processing %s: %s", symbol, exc)
            elif (
                trading_active
                and config.ENABLE_TRADING
                and not config.SHADOW_MODE
                and not permissions.can_open_trade(execution.exchange)
            ):
                logger.debug("Skipping entries: live trading permissions not granted")

            # MONITOREAR OPERACIONES ABIERTAS
            if config.SHADOW_MODE:
                realized = _process_shadow_positions()
                daily_profit += realized
            else:
                for op in list(all_open_trades()):
                    price = data.get_current_price_ticker(op["symbol"])
                    if not price:
                        continue
                    if op["side"] == "BUY":
                        profit = (price - op["entry_price"]) * op["quantity"]
                        close = price <= op["stop_loss"] or price >= op["take_profit"]
                    else:
                        profit = (op["entry_price"] - price) * op["quantity"]
                        close = price >= op["stop_loss"] or price <= op["take_profit"]

                    reason = None
                    if close:
                        reason = "TP" if profit >= 0 else "SL"
                    if reason is None and trade_manager.exceeded_max_duration(op):
                        close = True
                        reason = "MAX_DURATION"

                    if close:
                        expected = op["take_profit"] if (
                            (op["side"] == "BUY" and price >= op["take_profit"]) or
                            (op["side"] == "SELL" and price <= op["take_profit"])
                        ) else op["stop_loss"]
                        close_label = reason or ("TP" if profit >= 0 else "SL")
                        closed, exec_price, realized = close_existing_trade(
                            op,
                            close_label,
                        )
                        if not closed or exec_price is None or realized is None:
                            continue
                        slippage = exec_price - expected
                        logger.info(
                            "Closed %s at %.4f (target %.4f) slippage %.4f",
                            op["symbol"], exec_price, expected, slippage
                        )
                        if abs(slippage) > config.MAX_SLIPPAGE:
                            logger.warning("High slippage detected on %s: %.4f", op["symbol"], slippage)
                        daily_profit += realized

                        outcome = close_label
                        note = (
                            f"Closed {op['symbol']} {outcome} "
                            f"PnL {realized:.2f} Slippage {slippage:.4f}"
                        )
                        notify.send_telegram(note)
                        notify.send_discord(note)

            save_trades()  # Guarda el estado periódicamente
            update_trade_metrics(count_open_trades(), len(trade_manager.all_closed_trades()))

            if not trading_active and count_open_trades() == 0 and not standby_notified:
                logger.info("All positions closed after reaching daily limit")
                save_trades()
                standby_notified = True
            time.sleep(60)
        except KeyboardInterrupt:

            # On manual interrupt, cancel any pending orders and persist trades
            for order in execution.fetch_open_orders():
                sym = order.get("symbol", "").replace("/", "_").replace(":USDT", "")
                execution.cancel_order(order.get("id"), sym)
            save_trades()

            break
        except Exception as exc:
            logger.error("Loop error: %s", exc)
            time.sleep(10)

    logger.info("Trading loop exited")

def main() -> None:
    """Entry point that selects runtime mode and starts the bot."""

    args = parse_args()
    env_mode = config.BOT_MODE
    if args.mode:
        chosen = bot_mode.resolve_mode(args.mode, None)
    elif not args.no_interactive_menu and sys.stdin.isatty() and not env_mode:
        chosen = bot_mode.interactive_pick()
    else:
        chosen = bot_mode.resolve_mode(None, env_mode)

    bot_mode.apply_mode_to_config(chosen, config)

    if config.RUN_BACKTEST_ON_START:
        _run_backtest_startup(args)
        return


    run()


if __name__ == "__main__":
    main()
