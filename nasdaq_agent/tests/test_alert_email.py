from email.message import EmailMessage
from pathlib import Path

from agent import alert_email


ROOT = Path(__file__).resolve().parents[1]


class _DummySMTP:
    sent: list[EmailMessage] = []
    started_tls = False
    logged_in = False

    def __init__(self, host, port, timeout=10):
        self.host = host
        self.port = port
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def starttls(self):
        type(self).started_tls = True

    def login(self, username, password):
        type(self).logged_in = bool(username and password)

    def send_message(self, msg):
        type(self).sent.append(msg)


def test_alert_email_sends_redacted_smtp_message(monkeypatch):
    _DummySMTP.sent = []
    _DummySMTP.started_tls = False
    _DummySMTP.logged_in = False
    monkeypatch.setenv("ALERT_EMAIL_TO", "ops@example.com")
    monkeypatch.setenv("ALERT_EMAIL_FROM", "agent@example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "smtp-user")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-pass")
    monkeypatch.setenv("SMTP_STARTTLS", "true")
    monkeypatch.setattr(alert_email.smtplib, "SMTP", _DummySMTP)

    ok, err = alert_email.send_alert_email({
        "severity": "CRITICAL",
        "alert_type": "SCHWAB_AUTH",
        "source": "marketdata",
        "title": "Schwab MarketData re-authentication required",
        "message": "Refresh token was rejected.",
        "metadata": {"refresh_token": "secret", "http_code": 400},
    })

    assert ok, err
    assert _DummySMTP.started_tls
    assert _DummySMTP.logged_in
    assert len(_DummySMTP.sent) == 1
    msg = _DummySMTP.sent[0]
    assert msg["To"] == "ops@example.com"
    assert "CRITICAL" in msg["Subject"]
    body = msg.get_content()
    assert "SCHWAB_AUTH" in body
    assert "refresh_token: <redacted>" in body
    assert "secret" not in body


def test_system_alerts_and_compose_wire_operational_email():
    system_alerts = (ROOT / "agent/system_alerts.py").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "from agent.alert_email import is_configured, send_alert_email" in system_alerts
    assert "_maybe_send_email(alert_payload)" in system_alerts
    assert "ALERT_EMAIL_MIN_SEVERITY" in system_alerts
    assert "x-alert-env" in compose
    assert "ALERT_EMAIL_TO" in compose
    assert "SMTP_HOST" in compose
    assert "<<: [*db-env, *valkey-env, *schwab-env, *log-env, *alert-env]" in compose


def test_schwab_missing_token_paths_raise_system_alerts():
    token_service = (ROOT / "services/token_service.py").read_text(encoding="utf-8")
    schwab_auth = (ROOT / "agent/broker/schwab_auth.py").read_text(encoding="utf-8")

    assert "def _raise_missing_token_alert" in token_service
    assert '_raise_missing_token_alert("MarketData", "/schwab/auth/md")' in token_service
    assert '_raise_missing_token_alert("Trader", "/schwab/auth/at")' in token_service
    assert "missing_refresh_token" in schwab_auth
    assert "dedup_key=f\"SCHWAB_AUTH:{self.name.lower()}\"" in schwab_auth
