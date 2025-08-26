# tests/conftest.py
import os, socket, contextlib, pytest

# --- Bloquear red por defecto (evita cuelgues por timeouts HTTP/WS) ---
class _NetworkBlocked(Exception):
    pass


@contextlib.contextmanager
def _block_network():
    real_socket = socket.socket

    def guarded_socket(*a, **k):
        s = real_socket(*a, **k)
        try:
            real_connect = s.connect
        except AttributeError:
            return s

        def guarded_connect(addr):
            host, *_ = addr
            if host in ("127.0.0.1", "localhost", "::1"):
                return real_connect(addr)
            raise _NetworkBlocked(f"Network blocked: {addr}")

        try:
            s.connect = guarded_connect
        except AttributeError:
            pass
        return s

    socket.socket = guarded_socket
    try:
        yield
    finally:
        socket.socket = real_socket


@pytest.fixture(autouse=True, scope="session")
def _session_block_network():
    with _block_network():
        yield


# --- Entorno de test y tiempos m\xednimos ---
@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("ORDER_FILL_TIMEOUT", "1")   # evita esperas largas
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    yield


# --- Mock muy simple de requests.get/post (por si tu c\xf3digo los llama) ---
class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def mock_requests(monkeypatch):
    try:
        import requests
    except Exception:
        return
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200, {"ok": True}), raising=True)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(200, {"ok": True}), raising=True)


# --- Reset del estado de trades entre tests (si existe) ---
@pytest.fixture(autouse=True)
def reset_trade_state():
    try:
        from trading_bot import trade_manager as tm
        if hasattr(tm, "reset_state"):
            tm.reset_state()
    except Exception:
        pass
    yield

