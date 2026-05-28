import importlib.util
from pathlib import Path


def _load_gunicorn_conf(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    path = Path(__file__).resolve().parents[1] / "gunicorn.conf.py"
    spec = importlib.util.spec_from_file_location("gunicorn_conf_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backup_archive_probes_are_filtered(monkeypatch, tmp_path):
    conf = _load_gunicorn_conf(monkeypatch, tmp_path)
    msg = '172.18.0.1:50134 - "GET /v2/admin.zip HTTP/1.0" 404'
    assert conf._is_noise_probe_message(msg)


def test_backup_sql_probes_are_filtered(monkeypatch, tmp_path):
    conf = _load_gunicorn_conf(monkeypatch, tmp_path)
    msg = '172.18.0.1:50392 - "GET /v2/administrator.sql.gz HTTP/1.0" 404'
    assert conf._is_noise_probe_message(msg)


def test_deploy_archive_probes_are_filtered(monkeypatch, tmp_path):
    conf = _load_gunicorn_conf(monkeypatch, tmp_path)
    msg = '34.18.163.161 - - "GET /deploy/redis.tar.gz HTTP/1.1" 404'
    assert conf._is_noise_probe_message(msg)


def test_normal_404_is_kept(monkeypatch, tmp_path):
    conf = _load_gunicorn_conf(monkeypatch, tmp_path)
    msg = '172.18.0.1:50392 - "GET /watchlist/missing HTTP/1.0" 404'
    assert not conf._is_noise_probe_message(msg)


def test_health_200_is_filtered_by_default(monkeypatch, tmp_path):
    conf = _load_gunicorn_conf(monkeypatch, tmp_path)
    msg = '127.0.0.1:49078 - "GET /api/health HTTP/1.1" 200'
    assert conf._is_noise_probe_message(msg)


def test_health_filter_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("NASDAQ_SUPPRESS_HEALTH_ACCESS_LOGS", "0")
    conf = _load_gunicorn_conf(monkeypatch, tmp_path)
    msg = '127.0.0.1:49078 - "GET /api/health HTTP/1.1" 200'
    assert not conf._is_noise_probe_message(msg)
