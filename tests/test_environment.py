from robotics_stack.config.environment import dream_overrides


def test_environment_precedence_and_empty_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in ("RSTACK_DREAM_MODEL", "RSTACK_DREAM_SOLVER", "RSTACK_DREAM_DRY_RUN"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "RSTACK_DREAM_MODEL=local\nRSTACK_DREAM_SOLVER=\nRSTACK_DREAM_DRY_RUN=false\n"
    )
    assert dream_overrides()["checkpoint"] == "local"
    assert "ik_solve" not in dream_overrides()
    assert dream_overrides()["dry_run"] is False
    monkeypatch.setenv("RSTACK_DREAM_MODEL", "exported")
    assert dream_overrides()["checkpoint"] == "exported"
