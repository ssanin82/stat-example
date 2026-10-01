from __future__ import annotations

import os
import sys

import pytest

from app.env_bootstrap import (
    ensure_env_bootstrapped,
    env_file_startup_log_line,
    extract_env_file_from_argv,
    reset_env_bootstrap_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_bootstrap():
    reset_env_bootstrap_for_tests()
    yield
    reset_env_bootstrap_for_tests()


def test_loads_vars_from_file(tmp_path):
    f = tmp_path / "p.env"
    f.write_text("MM_BOOT_TEST_FROM_FILE=1\n", encoding="utf-8")
    old = os.environ.pop("MM_BOOT_TEST_FROM_FILE", None)
    try:
        argv = [sys.argv[0], "--env-file", str(f)]
        ensure_env_bootstrapped(argv=argv)
        assert os.environ.get("MM_BOOT_TEST_FROM_FILE") == "1"
        assert "loaded from" in env_file_startup_log_line()
    finally:
        if old is not None:
            os.environ["MM_BOOT_TEST_FROM_FILE"] = old
        else:
            os.environ.pop("MM_BOOT_TEST_FROM_FILE", None)


def test_process_env_overrides_file(tmp_path):
    f = tmp_path / "p.env"
    f.write_text("MM_BOOT_TEST_OVERRIDE=file\n", encoding="utf-8")
    os.environ["MM_BOOT_TEST_OVERRIDE"] = "env"
    try:
        ensure_env_bootstrapped(argv=[sys.argv[0], "--env-file", str(f)])
        assert os.environ.get("MM_BOOT_TEST_OVERRIDE") == "env"
    finally:
        os.environ.pop("MM_BOOT_TEST_OVERRIDE", None)


def test_precedence_cli_over_app_env_file(tmp_path):
    a = tmp_path / "a.env"
    b = tmp_path / "b.env"
    a.write_text("MM_BOOT_TEST_PRECEDENCE=a\n", encoding="utf-8")
    b.write_text("MM_BOOT_TEST_PRECEDENCE=b\n", encoding="utf-8")
    old = os.environ.pop("MM_BOOT_TEST_PRECEDENCE", None)
    old_app = os.environ.pop("APP_ENV_FILE", None)
    try:
        os.environ["APP_ENV_FILE"] = str(a)
        argv = [sys.argv[0], "--env-file", str(b)]
        ensure_env_bootstrapped(argv=argv)
        assert os.environ.get("MM_BOOT_TEST_PRECEDENCE") == "b"
        line = env_file_startup_log_line()
        assert str(b.resolve()) in line
    finally:
        if old is not None:
            os.environ["MM_BOOT_TEST_PRECEDENCE"] = old
        if old_app is not None:
            os.environ["APP_ENV_FILE"] = old_app
        else:
            os.environ.pop("APP_ENV_FILE", None)


def test_app_env_file_when_no_argv(tmp_path):
    f = tmp_path / "only.env"
    f.write_text("MM_BOOT_APP_ONLY=1\n", encoding="utf-8")
    old = os.environ.pop("MM_BOOT_APP_ONLY", None)
    old_app = os.environ.pop("APP_ENV_FILE", None)
    try:
        os.environ["APP_ENV_FILE"] = str(f)
        ensure_env_bootstrapped(argv=None)
        assert os.environ.get("MM_BOOT_APP_ONLY") == "1"
    finally:
        if old is not None:
            os.environ["MM_BOOT_APP_ONLY"] = old
        if old_app is not None:
            os.environ["APP_ENV_FILE"] = old_app
        else:
            os.environ.pop("APP_ENV_FILE", None)


def test_extract_env_file_from_argv_strips_flags():
    cli, rest = extract_env_file_from_argv(
        ["prog", "--env-file", "/x/y.env", "extra"]
    )
    assert cli == "/x/y.env"
    assert rest == ["prog", "extra"]


def test_extract_env_file_equals_form():
    cli, rest = extract_env_file_from_argv(["prog", "--env-file=/p.env"])
    assert cli == "/p.env"
    assert rest == ["prog"]
