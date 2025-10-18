from __future__ import annotations

from trading_bot import bot


def test_normalize_dashboard_host_defaults() -> None:
    assert bot._normalize_dashboard_host("0.0.0.0") == "127.0.0.1"
    assert bot._normalize_dashboard_host("::") == "127.0.0.1"
    assert bot._normalize_dashboard_host("example.com") == "example.com"
    assert bot._normalize_dashboard_host("") == "127.0.0.1"


def test_dashboard_url_formats_ipv6() -> None:
    assert bot._dashboard_url("0.0.0.0", 1234) == "http://127.0.0.1:1234"
    assert bot._dashboard_url("2001:db8::1", 8080) == "http://[2001:db8::1]:8080"
    assert bot._dashboard_url("https://custom", 80) == "https://custom"


def test_launch_dashboard_browser_opens_when_available(monkeypatch) -> None:
    attempts: list[tuple[str, int]] = []

    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_create_connection(address, timeout=0):
        attempts.append(address)
        return DummySocket()

    opened: dict[str, object] = {}

    def fake_open(url, *, new=0, autoraise=True):
        opened["url"] = url
        opened["new"] = new
        opened["autoraise"] = autoraise
        return True

    monkeypatch.setattr(bot.socket, "create_connection", fake_create_connection)
    monkeypatch.setattr(bot.webbrowser, "open", fake_open)

    bot._launch_dashboard_browser("0.0.0.0", 5000, attempts=1)

    assert attempts == [("127.0.0.1", 5000)]
    assert opened["url"] == "http://127.0.0.1:5000"
    assert opened["new"] == 1
    assert opened["autoraise"] is True


def test_launch_dashboard_browser_fallback(monkeypatch) -> None:
    calls: dict[str, int | str] = {"open": 0, "sleep": 0}

    def fake_create_connection(*_args, **_kwargs):
        raise OSError("offline")

    def fake_open(url, *, new=0, autoraise=True):
        calls["open"] = calls.get("open", 0) + 1
        calls["url"] = url
        calls["new"] = new
        calls["autoraise"] = autoraise
        return False

    def fake_sleep(_seconds):
        calls["sleep"] = calls.get("sleep", 0) + 1

    monkeypatch.setattr(bot.socket, "create_connection", fake_create_connection)
    monkeypatch.setattr(bot.webbrowser, "open", fake_open)
    monkeypatch.setattr(bot.time, "sleep", fake_sleep)

    bot._launch_dashboard_browser("127.0.0.1", 6000, attempts=2, delay=0)

    assert calls["open"] == 1
    assert calls["url"] == "http://127.0.0.1:6000"
    assert calls["new"] == 1
    assert calls["autoraise"] is True
    assert calls["sleep"] == 2


def test_should_auto_launch_dashboard_defaults(monkeypatch) -> None:
    class FakeStdin:
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("AUTO_OPEN_DASHBOARD", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(bot.sys, "stdin", FakeStdin())

    assert bot._should_auto_launch_dashboard() is True

    class NoTty:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(bot.sys, "stdin", NoTty())
    assert bot._should_auto_launch_dashboard() is False


def test_should_auto_launch_dashboard_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AUTO_OPEN_DASHBOARD", "0")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(bot.sys, "stdin", None)

    assert bot._should_auto_launch_dashboard() is False

    monkeypatch.setenv("AUTO_OPEN_DASHBOARD", "yes")
    assert bot._should_auto_launch_dashboard() is True
