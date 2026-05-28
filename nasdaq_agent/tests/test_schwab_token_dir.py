import json


def test_token_manager_uses_configured_token_dir(monkeypatch, tmp_path):
    from agent.broker import schwab_auth

    token_dir = tmp_path / "tokens"
    monkeypatch.setenv("SCHWAB_TOKEN_DIR", str(token_dir))
    monkeypatch.setattr(schwab_auth, "_BACKUP_DIR", tmp_path / "backup")

    mgr = schwab_auth._TokenManager(
        name="Test",
        client_id_env="SCHWAB_TEST_ID",
        client_secret_env="SCHWAB_TEST_SECRET",
        token_filename="test_tokens.json",
    )
    mgr._tokens = {"access_token": "access", "refresh_token": "refresh"}
    mgr._save()

    assert mgr._token_path == token_dir / "test_tokens.json"
    assert json.loads((token_dir / "test_tokens.json").read_text())["access_token"] == "access"


def test_token_manager_migrates_legacy_data_token(monkeypatch, tmp_path):
    from agent.broker import schwab_auth

    legacy_dir = tmp_path / "data"
    token_dir = tmp_path / "tokens"
    legacy_dir.mkdir()
    (legacy_dir / "test_tokens.json").write_text(json.dumps({"access_token": "old"}))

    monkeypatch.setenv("SCHWAB_TOKEN_DIR", str(token_dir))
    monkeypatch.setattr(schwab_auth, "_DEFAULT_TOKEN_DIR", legacy_dir)
    monkeypatch.setattr(schwab_auth, "_BACKUP_DIR", tmp_path / "backup")

    mgr = schwab_auth._TokenManager(
        name="Test",
        client_id_env="SCHWAB_TEST_ID",
        client_secret_env="SCHWAB_TEST_SECRET",
        token_filename="test_tokens.json",
    )

    assert mgr._load_from_disk()["access_token"] == "old"
    assert json.loads((token_dir / "test_tokens.json").read_text())["access_token"] == "old"
