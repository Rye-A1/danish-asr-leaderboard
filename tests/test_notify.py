"""Tests for the webhook notifier (stdlib only; no network)."""
import danish_asr_leaderboard.notify as notify_mod
from danish_asr_leaderboard.notify import get_webhook_url, notify, send_mattermost

_KEY = "MATTERMOST_WEBHOOK_URL"


def test_explicit_url_wins(monkeypatch):
    monkeypatch.delenv(_KEY, raising=False)
    assert get_webhook_url("https://hook.example/x") == "https://hook.example/x"


def test_env_var_resolves(monkeypatch):
    monkeypatch.setenv(_KEY, "https://hook.example/from-env")
    assert get_webhook_url() == "https://hook.example/from-env"


def test_unset_returns_empty(monkeypatch, tmp_path):
    # Isolate from any real .env: no env var, empty cwd, repo-root pointed at tmp.
    monkeypatch.delenv(_KEY, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(notify_mod, "_REPO_ROOT", tmp_path)
    assert get_webhook_url() == ""


def test_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv(_KEY, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(notify_mod, "_REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text(f'{_KEY}="https://hook.example/from-dotenv"\n')
    assert get_webhook_url() == "https://hook.example/from-dotenv"


def test_send_and_notify_noop_when_unset(monkeypatch, tmp_path):
    # Empty URL must never raise or attempt a request.
    send_mattermost("", "hello")
    monkeypatch.delenv(_KEY, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(notify_mod, "_REPO_ROOT", tmp_path)
    notify("hello")  # resolves to "" → no-op
