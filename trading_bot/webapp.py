from __future__ import annotations

import csv
import json
import logging
import os
import queue
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

USE_EVENTLET = os.getenv("USE_EVENTLET", "1") == "1"
_EVENTLET_IMPORT_ERROR: str | None = None

if USE_EVENTLET:
    try:  # pragma: no cover - optional dependency en producción
        import eventlet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - eventlet opcional
        eventlet = None  # type: ignore[assignment]
        USE_EVENTLET = False
        _EVENTLET_IMPORT_ERROR = str(exc)
else:
    eventlet = None  # type: ignore[assignment]

try:
    from flask import (
        Flask,
        Response,
        jsonify,
        redirect,
        render_template,
        request,
        send_from_directory,
        url_for,
    )
except ImportError:  # Flask not installed
    Flask = None

try:
    from flask_socketio import SocketIO, disconnect, emit
except ImportError:  # SocketIO optional
    SocketIO = None
    disconnect = None  # type: ignore[assignment]
    emit = None  # type: ignore[assignment]

try:
    from flask_login import (
        LoginManager,
        current_user,
        login_required,
        login_user,
        logout_user,
    )
except ImportError:  # pragma: no cover - optional dependency
    LoginManager = None

    def login_required(func):  # type: ignore[misc]
        return func

    def login_user(*_args, **_kwargs):  # pragma: no cover - safety fallback
        raise RuntimeError("Flask-Login no está instalado")

    def logout_user(*_args, **_kwargs):  # pragma: no cover - safety fallback
        raise RuntimeError("Flask-Login no está instalado")

    class _FallbackUser:
        is_authenticated = False
        username = ""

    current_user = _FallbackUser()  # type: ignore[assignment]

from trading_bot import config
from trading_bot.trade_manager import (
    all_closed_trades,
    all_open_trades,
    close_trade_full,
    close_trade_partial,
    get_open_trade,
)
from trading_bot import data
from trading_bot.history import HISTORY_FILE, FIELDS as HISTORY_FIELDS
from trading_bot.authentication import authenticate, ensure_default_admin, has_any_user, load_user
from trading_bot.db import init_db, remove_session

logger = logging.getLogger(__name__)

if _EVENTLET_IMPORT_ERROR:
    logger.warning(
        "USE_EVENTLET=1 pero eventlet no está instalado: %s. El dashboard usará threading.",
        _EVENTLET_IMPORT_ERROR,
    )

ASYNC_MODE = "eventlet" if USE_EVENTLET else "threading"

if Flask:
    app = Flask(__name__)

    secret_key = config.DASHBOARD_SECRET_KEY.strip()
    if not secret_key:
        secret_key = os.getenv("FLASK_SECRET_KEY", "") or os.getenv("SECRET_KEY", "")
    if not secret_key:
        secret_key = os.urandom(32).hex()
        logger.warning(
            "No se proporcionó DASHBOARD_SECRET_KEY. Se generó uno temporal en memoria."
        )
    app.secret_key = secret_key

    init_db()
    _AUTH_ENABLED = False
    _default_admin_created = False
    try:
        _default_admin_created = ensure_default_admin()
    except Exception:  # pragma: no cover - setup guard
        logger.exception("Error al inicializar el usuario administrador predeterminado")
    try:
        _existing_users = has_any_user()
    except Exception:  # pragma: no cover - setup guard
        logger.exception("No se pudo verificar usuarios existentes del dashboard")
        _existing_users = False

    socketio = (
        SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode=ASYNC_MODE,
            ping_timeout=30,
            ping_interval=15,
        )
        if SocketIO
        else None
    )

    _connected_clients = 0
    _connected_lock = threading.Lock()

    def _inc_clients() -> None:
        global _connected_clients
        with _connected_lock:
            _connected_clients += 1

    def _dec_clients() -> None:
        global _connected_clients
        with _connected_lock:
            _connected_clients = max(0, _connected_clients - 1)

    _sse_clients: set[queue.Queue[str]] = set()
    _sse_lock = threading.Lock()

    def _sse_register() -> queue.Queue[str]:
        client_queue: queue.Queue[str] = queue.Queue()
        with _sse_lock:
            _sse_clients.add(client_queue)
        return client_queue

    def _sse_unregister(client_queue: queue.Queue[str]) -> None:
        with _sse_lock:
            _sse_clients.discard(client_queue)

    def _push_to_sse(event: str, payload: Any) -> None:
        try:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            logger.debug("No se pudo serializar payload para SSE: %s", event)
            return

        message = f"event: {event}\ndata: {data}\n\n"
        with _sse_lock:
            stale_clients: list[queue.Queue[str]] = []
            for client_queue in _sse_clients:
                try:
                    client_queue.put_nowait(message)
                except Exception:
                    stale_clients.append(client_queue)
            for client_queue in stale_clients:
                _sse_clients.discard(client_queue)

    if LoginManager:
        _AUTH_ENABLED = bool(
            config.DASHBOARD_REQUIRE_AUTH or _default_admin_created or _existing_users
        )
        login_manager = LoginManager(app)
        login_manager.login_view = "login"

        @login_manager.user_loader
        def _load_user(user_id: str):
            return load_user(user_id)

        if _AUTH_ENABLED:
            @login_manager.unauthorized_handler
            def _unauthorized():
                if request.path.startswith("/api/"):
                    return jsonify({"ok": False, "error": "authentication_required"}), 401
                next_url = request.full_path if request.method == "GET" else request.path
                if next_url.endswith("?"):
                    next_url = next_url[:-1]
                if not _is_safe_redirect(next_url):
                    next_url = url_for("index")
                return redirect(url_for("login", next=next_url))
        else:
            if config.DASHBOARD_REQUIRE_AUTH:
                logger.warning(
                    "DASHBOARD_REQUIRE_AUTH=1 pero no hay usuarios configurados. El acceso no se protegerá hasta crear usuarios."
                )
            logger.info("Autenticación del dashboard deshabilitada (sin usuarios registrados).")
    else:
        _AUTH_ENABLED = False

    if socketio:
        @socketio.on("connect", namespace="/ws")
        def _ws_connect():  # pragma: no cover - relies on Socket.IO runtime
            if (
                _AUTH_ENABLED
                and disconnect
                and not getattr(current_user, "is_authenticated", False)
            ):
                disconnect()
                return

            logger.info("WS connect: %s", getattr(request, "sid", "?"))
            _inc_clients()
            emit("bot_status", {"trading_active": bool(getattr(config, "AUTO_TRADE", True))})
            emit("trades_refresh", _trades_with_metrics())

        @socketio.on("disconnect", namespace="/ws")
        def _ws_disconnect():  # pragma: no cover - relies on Socket.IO runtime
            logger.info("WS disconnect: %s", getattr(request, "sid", "?"))
            _dec_clients()

    @app.teardown_appcontext
    def _cleanup_session(exception: Exception | None) -> None:  # pragma: no cover - glue code
        remove_session()

    _trade_locks: dict[str, threading.Lock] = {}
    _locks_lock = threading.Lock()
    _session_lock = threading.Lock()
    _session_started_at = datetime.utcnow()
    _session_has_positions = False
    _trading_state_lock = threading.Lock()

    _TRADING_STATE_PATH = Path(getattr(config, "__file__", __file__)).with_name(
        "auto_trade_state.json"
    )

    def _write_trading_state(active: bool) -> None:
        """Persist the AUTO_TRADE flag so it survives restarts."""

        path = _TRADING_STATE_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(".tmp")
            payload = {
                "auto_trade": bool(active),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
        except OSError as exc:  # pragma: no cover - disk issues are unlikely in tests
            logger.warning("No se pudo persistir el estado de AUTO_TRADE: %s", exc)

    def _apply_trading_state(active: bool, *, persist: bool = False) -> bool:
        """Update config flags and optionally persist them to disk."""

        active_flag = bool(active)
        with _trading_state_lock:
            config.AUTO_TRADE = active_flag
            config.MAINTENANCE = not active_flag
            if persist:
                _write_trading_state(active_flag)
        return active_flag

    def _load_trading_state() -> bool:
        """Load AUTO_TRADE from disk if available and keep it consistent."""

        stored = bool(getattr(config, "AUTO_TRADE", True))
        path = _TRADING_STATE_PATH
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("No se pudo leer el estado persistido de AUTO_TRADE: %s", exc)
            else:
                if isinstance(raw, dict) and "auto_trade" in raw:
                    stored = bool(raw["auto_trade"])
        return _apply_trading_state(stored, persist=True)

    _load_trading_state()

    def _get_trade_lock(trade_id: str) -> threading.Lock:
        with _locks_lock:
            if trade_id not in _trade_locks:
                _trade_locks[trade_id] = threading.Lock()
            return _trade_locks[trade_id]

    def _session_identifier(has_positions: bool) -> str:
        global _session_started_at, _session_has_positions
        with _session_lock:
            if not has_positions and _session_has_positions:
                _session_started_at = datetime.utcnow()
            elif _session_started_at is None:
                _session_started_at = datetime.utcnow()
            _session_has_positions = has_positions
            started = _session_started_at or datetime.utcnow()
            return started.replace(microsecond=0).isoformat() + "Z"

    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _trades_with_metrics() -> list[dict[str, Any]]:
        trades: list[dict[str, Any]] = []
        for trade in all_open_trades():
            sym = trade.get("symbol")
            entry = _coerce_float(trade.get("entry_price", 0))
            qty = abs(
                _coerce_float(
                    trade.get("quantity_remaining", trade.get("quantity", 0))
                )
            )
            side = str(trade.get("side", "BUY")).upper()
            current_price_raw = data.get_current_price_ticker(sym)
            if current_price_raw is None:
                current_price_raw = entry
            current_price = _coerce_float(current_price_raw)
            leverage = 1.0
            try:
                leverage = float(trade.get("leverage") or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0
            if leverage <= 0:
                leverage = 1.0
            invested_value = 0.0
            if entry > 0 and qty > 0:
                invested_value = abs(entry * qty) / leverage
            if side == "BUY":
                pnl = (current_price - entry) * qty / leverage
            else:
                pnl = (entry - current_price) * qty / leverage
            if invested_value > 0:
                pnl = max(pnl, -invested_value)
            row = dict(trade)
            row["entry_price"] = entry
            row["quantity"] = qty
            row["quantity_remaining"] = qty
            row["current_price"] = current_price
            row["pnl_unrealized"] = pnl
            row["invested_value"] = invested_value
            row["realized_pnl"] = _coerce_float(trade.get("realized_pnl"))
            row["leverage"] = leverage
            trades.append(row)
        return trades

    def _realized_aggregates() -> tuple[float, float, int, int]:
        """Compute realized balance, PnL and win/loss counts from closed trades."""

        realized_balance = 0.0
        realized_pnl_total = 0.0
        winners = 0
        losers = 0

        for trade in all_closed_trades():
            realized = _coerce_float(trade.get("realized_pnl") or trade.get("profit"))
            realized_pnl_total += realized
            if realized > 0:
                realized_balance += realized
                winners += 1
            elif realized < 0:
                losers += 1

        return realized_balance, realized_pnl_total, winners, losers

    def _is_safe_redirect(target: str) -> bool:
        if not target:
            return False
        parsed = urlparse(target)
        if parsed.netloc:
            return False
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            return False
        return not target.startswith("//")

    def _service_links() -> list[dict[str, str]]:
        raw = getattr(config, "EXTERNAL_SERVICE_LINKS", "")
        if not raw:
            return []
        tokens = raw.replace(";", "\n").splitlines()
        links: list[dict[str, str]] = []
        for token in tokens:
            entry = token.strip()
            if not entry:
                continue
            if "|" in entry:
                label, url = entry.split("|", 1)
            elif "," in entry:
                label, url = entry.split(",", 1)
            else:
                label, url = entry, entry
            label = label.strip() or url.strip()
            url = url.strip()
            if not url:
                continue
            links.append({"label": label, "url": url})
        return links

    def _maybe_login_required(func):
        if not _AUTH_ENABLED:
            return func
        return login_required(func)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        safe_next = request.args.get("next") or ""
        if not _is_safe_redirect(safe_next):
            safe_next = ""
        if not _AUTH_ENABLED:
            return redirect(safe_next or url_for("index"))
        if getattr(current_user, "is_authenticated", False):
            return redirect(safe_next or url_for("index"))
        if request.method == "POST":
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""
            raw_next = request.args.get("next") or request.form.get("next") or ""
            next_url = raw_next if _is_safe_redirect(raw_next) else url_for("index")
            user = authenticate(username, password)
            if user:
                login_user(user)
                return redirect(next_url)
            error = "Credenciales inválidas"
        else:
            error = None
        return render_template("login.html", error=error, next=safe_next)

    @app.route("/logout")
    @_maybe_login_required
    def logout():
        if not _AUTH_ENABLED:
            return redirect(url_for("index"))
        logout_user()
        return redirect(url_for("login"))

    @app.route("/")
    @_maybe_login_required
    def index():
        gateway_base = getattr(config, "DASHBOARD_GATEWAY_BASE", "").strip()
        if not gateway_base:
            gateway_base = request.url_root.rstrip("/")

        analytics_graphql = getattr(config, "ANALYTICS_GRAPHQL_URL", "").strip()

        ai_endpoint = getattr(config, "AI_ASSISTANT_URL", "").strip()
        socket_base = getattr(config, "DASHBOARD_SOCKET_BASE", "")
        socket_path = getattr(config, "DASHBOARD_SOCKET_PATH", "")
        return render_template(
            "index.html",
            current_year=datetime.utcnow().year,
            api_base=gateway_base,
            analytics_graphql_url=analytics_graphql,
            ai_api_url=ai_endpoint,
            socket_base=socket_base,
            socket_path=socket_path,
            service_links=_service_links(),
        )

    @app.route("/manifest.json")
    def manifest():
        return send_from_directory(app.static_folder, "manifest.json")

    @app.route("/service-worker.js")
    def service_worker():
        return send_from_directory(app.static_folder, "service-worker.js")

    @app.get("/api/health")
    def api_health():
        return jsonify(
            {
                "ok": True,
                "socketio": bool(socketio is not None),
                "auto_trade": bool(getattr(config, "AUTO_TRADE", True)),
                "ws_clients": _connected_clients,
            }
        )

    @app.route("/api/trades")
    @_maybe_login_required
    def api_trades():
        """Return open trades with current price and unrealized PnL."""
        return jsonify(_trades_with_metrics())

    @app.post("/api/toggle-trading")
    @_maybe_login_required
    def api_toggle_trading():
        """Permite pausar o reanudar el trading automático desde el dashboard."""

        current = bool(getattr(config, "AUTO_TRADE", True))
        new_status = _apply_trading_state(not current, persist=True)
        state_label = "habilitado" if new_status else "pausado"
        logger.warning("Trading %s por solicitud vía dashboard", state_label)
        payload = {
            "ok": True,
            "trading_active": new_status,
            "status": state_label,
            "persisted": True,
        }
        _emit("bot_status", {"trading_active": new_status})
        return jsonify(payload)

    def _push_to_webview(event: str, payload: Any) -> None:
        """Mirror dashboard events to the desktop bridge when available."""

        try:
            from . import desktop  # noqa: WPS433 - optional dependency
        except Exception:
            return

        evaluator = getattr(desktop, "eval_js", None)
        if not callable(evaluator):
            return

        try:
            js_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            logger.debug("No se pudo serializar payload para PyWebview: %s", event)
            return

        script = (
            "window.__pybridge && window.__pybridge.emit("
            f"{json.dumps(event)}, {js_payload});"
        )
        try:
            evaluator(script)
        except Exception:
            logger.debug("Fallo al emitir evento %s hacia PyWebview", event)

    def _emit(event: str, payload: Any) -> None:
        if socketio:
            socketio.emit(event, payload, namespace="/ws")
        _push_to_sse(event, payload)
        _push_to_webview(event, payload)

    def broadcast_trades_refresh() -> None:
        """Emit a trades refresh event if Socket.IO is active."""

        _emit("trades_refresh", _trades_with_metrics())

    @app.get("/events")
    def sse_stream():
        def generate():
            client_queue = _sse_register()
            try:
                initial_status = {"trading_active": bool(getattr(config, "AUTO_TRADE", True))}
                yield "event: bot_status\n"
                yield f"data: {json.dumps(initial_status, ensure_ascii=False, separators=(",", ":"))}\n\n"
                yield "event: trades_refresh\n"
                yield f"data: {json.dumps(_trades_with_metrics(), ensure_ascii=False, separators=(",", ":"))}\n\n"
                while True:
                    chunk = client_queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                _sse_unregister(client_queue)

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        return Response(generate(), mimetype="text/event-stream", headers=headers)

    @app.post("/api/trades/<trade_id>/close")
    @_maybe_login_required
    def api_close_trade(trade_id: str):
        lock = _get_trade_lock(trade_id)
        with lock:
            trade = get_open_trade(trade_id)
            if not trade:
                closed = close_trade_full(trade_id)
                if not closed:
                    return jsonify({"ok": False, "error": "Trade no encontrado"}), 404
                return jsonify({"ok": True, "trade": closed, "note": "ya estaba cerrado"})
            payload = request.get_json(silent=True) or {}
            reason = payload.get("reason", "manual_close")
            result = close_trade_full(trade_id, reason=reason)
            if not result:
                return jsonify({"ok": False, "error": "No se pudo cerrar la operación"}), 400
        _emit("trade_closed", {"trade": result})
        _emit("trades_refresh", _trades_with_metrics())
        return jsonify({"ok": True, "trade": result})

    @app.post("/api/trades/<trade_id>/close-partial")
    @_maybe_login_required
    def api_close_trade_partial(trade_id: str):
        lock = _get_trade_lock(trade_id)
        with lock:
            trade = get_open_trade(trade_id)
            if not trade:
                return jsonify({"ok": False, "error": "Trade no encontrado o ya cerrado"}), 404

            data_payload = request.get_json(silent=True) or {}
            qty = data_payload.get("qty")
            pct = data_payload.get("percent")
            reason = data_payload.get("reason", "manual_partial")

            if qty is None and pct is None:
                return jsonify({"ok": False, "error": "Debes informar qty o percent"}), 400

            remaining = _coerce_float(trade.get("quantity_remaining", trade.get("quantity", 0)))
            if pct is not None:
                try:
                    pct_val = float(pct)
                except (TypeError, ValueError):
                    return jsonify(
                        {"ok": False, "error": "percent inválido (usa un número real)"}
                    ), 400
                if pct_val <= 0 or pct_val > 100:
                    return jsonify(
                        {
                            "ok": False,
                            "error": "percent fuera de rango permitido (1-100)",
                        }
                    ), 400
                qty = remaining * (pct_val / 100.0)

            try:
                qty_val = float(qty)
            except (TypeError, ValueError):
                return jsonify(
                    {
                        "ok": False,
                        "error": "qty inválido (usa un número mayor que 0)",
                    }
                ), 400
            if qty_val <= 0:
                return jsonify(
                    {"ok": False, "error": "qty debe ser mayor que 0"}
                ), 400
            if qty_val > remaining + 1e-12:
                return jsonify(
                    {
                        "ok": False,
                        "error": "qty excede la cantidad restante"
                        f" ({remaining:.6f})",
                    }
                ), 400

            try:
                result = close_trade_partial(trade_id, quantity=qty_val, reason=reason)
            except ValueError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            if not result:
                return jsonify({"ok": False, "error": "No se pudo cerrar parcialmente"}), 400
        _emit("trade_updated", {"trade": result})
        _emit("trades_refresh", _trades_with_metrics())
        return jsonify({"ok": True, "trade": result})

    @app.route("/api/summary")
    @_maybe_login_required
    def api_summary():
        """Aggregate metrics that power the dashboard widgets."""
        trades = _trades_with_metrics()
        total_positions = len(trades)
        total_quantity = sum(abs(t["quantity_remaining"]) for t in trades)
        total_invested = sum(t["invested_value"] for t in trades)
        total_exposure = sum(
            abs(t["quantity_remaining"] * t.get("current_price", 0.0)) for t in trades
        )
        unrealized_pnl = sum(t["pnl_unrealized"] for t in trades)
        open_winners = sum(1 for t in trades if t["pnl_unrealized"] > 0)
        open_losers = sum(1 for t in trades if t["pnl_unrealized"] < 0)
        (
            realized_balance,
            realized_pnl_total,
            closed_winners,
            closed_losers,
        ) = _realized_aggregates()
        session_id = _session_identifier(total_positions > 0)

        per_symbol: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "symbol": "",
                "positions": 0,
                "quantity": 0.0,
                "exposure": 0.0,
                "unrealized_pnl": 0.0,
                "invested_value": 0.0,
            }
        )

        for trade in trades:
            sym = str(trade.get("symbol", "")).upper()
            entry = per_symbol[sym]
            entry["symbol"] = sym
            entry["positions"] += 1
            entry["quantity"] += abs(trade["quantity_remaining"])
            entry["exposure"] += abs(
                trade["quantity_remaining"] * trade.get("current_price", 0.0)
            )
            entry["unrealized_pnl"] += trade["pnl_unrealized"]
            entry["invested_value"] += trade["invested_value"]

        per_symbol_list = sorted(
            per_symbol.values(),
            key=lambda item: item["exposure"],
            reverse=True,
        )

        win_rate: float | None = None
        total_winners = open_winners + closed_winners
        total_losers = open_losers + closed_losers
        deciding = total_winners + total_losers
        if deciding:
            win_rate = total_winners / deciding

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_positions": total_positions,
            "total_exposure": total_exposure,
            "total_invested": total_invested,
            "total_quantity": total_quantity,
            "unrealized_pnl": unrealized_pnl,
            "realized_balance": realized_balance,
            "realized_pnl": realized_pnl_total,
            "realized_pnl_total": realized_pnl_total,
            "total_pnl": unrealized_pnl + realized_pnl_total,
            "winning_positions": open_winners,
            "losing_positions": open_losers,
            "win_rate": win_rate,
            "win_rate_samples": deciding,
            "win_rate_winners": total_winners,
            "win_rate_losers": total_losers,
            "per_symbol": per_symbol_list,
            "session_id": session_id,
            "trading_active": bool(
                getattr(config, "AUTO_TRADE", True)
                and not getattr(config, "MAINTENANCE", False)
            ),
        }
        trading_mode = str(getattr(config, "TRADING_MODE", "live")).lower()
        payload["trading_mode"] = trading_mode
        payload["paper_trading"] = trading_mode != "live"
        return jsonify(payload)

    @app.route("/api/history")
    @_maybe_login_required
    def api_history():
        """Return the latest closed trades from the CSV history file."""
        try:
            limit = int(request.args.get("limit", 50))
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(limit, 200))

        rows: list[dict[str, Any]] = []
        if HISTORY_FILE.exists():
            with HISTORY_FILE.open(newline="") as handle:
                reader = list(csv.DictReader(handle, fieldnames=HISTORY_FIELDS))
            if reader:
                # The first row might be the header if DictReader didn't skip it
                if reader[0] and reader[0][HISTORY_FIELDS[0]] == HISTORY_FIELDS[0]:
                    reader = reader[1:]
                rows = reader[-limit:]

        formatted_rows: list[dict[str, Any]] = []
        for row in reversed(rows):
            formatted_rows.append(
                {
                    key: row.get(key, "")
                    for key in (
                        "symbol",
                        "side",
                        "quantity",
                        "entry_price",
                        "exit_price",
                        "profit",
                        "open_time",
                        "close_time",
                    )
                }
            )

        return jsonify(formatted_rows)

    def start_dashboard(host: str, port: int):
        """Run the Flask dashboard in real-time with trade data."""
        if socketio:
            socketio.run(app, host=host, port=port)
        else:
            app.run(host=host, port=port)
else:
    socketio = None

    def _push_to_sse(_event: str, _payload: Any) -> None:
        return None

    def broadcast_trades_refresh() -> None:
        """Fallback helper when the dashboard is unavailable."""

        return None

    def start_dashboard(host: str, port: int):
        raise ImportError("Flask is required to run the dashboard")
