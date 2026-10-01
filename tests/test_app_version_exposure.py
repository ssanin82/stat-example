"""Pin the FastAPI version exposure so a future refactor can't silently drop it.

When the operator pulls ``/openapi.json`` or ``/docs`` from a running
deployment, ``info.version`` must equal the current ``app.__version__``.
This lets every snapshot the operator captures be tied back to the
exact build — critical for rollback decisions when a change is
suspected of breaking production.

Two assertions:

  1. ``info.version`` on the generated OpenAPI spec matches
     ``app.__version__``. Guards against someone refactoring
     ``create_app()`` and dropping the ``version=`` kwarg on the
     ``FastAPI(...)`` call.
  2. ``app.__version__`` conforms to a simple ``MAJOR.MINOR.PATCH``
     shape. Cheap future-proofing against "pasted a git SHA / dirty
     build suffix" regressions.

Tests call ``create_app()`` directly — that's the production factory
``app/main.py::create_app()`` — so the assertion exercises the real
FastAPI kwargs, not a reconstructed test-only app.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

import app as app_pkg
from app.main import create_app


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def test_openapi_info_version_matches_app_version() -> None:
    """GET ``/openapi.json`` and confirm FastAPI is publishing
    ``app.__version__`` — not the framework default, not a blank,
    not a stale literal. The route is unauthenticated and always on."""
    fastapi_app = create_app()
    client = TestClient(fastapi_app)
    r = client.get("/openapi.json")
    assert r.status_code == 200, r.text
    info = r.json().get("info", {})
    assert info.get("version") == app_pkg.__version__, (
        f"openapi.json info.version={info.get('version')!r} does not match "
        f"app.__version__={app_pkg.__version__!r} — someone likely dropped "
        f"the `version=` kwarg from the FastAPI(...) call in create_app()."
    )


def test_app_version_is_semver_shape() -> None:
    """``__version__`` must be a plain MAJOR.MINOR.PATCH triple —
    no pre-release / build-metadata suffixes for now. Keeps the string
    parseable by external tooling (CI, deploy-host labels, etc.) without
    extra handling. Loosen this later if we adopt richer SemVer."""
    v = app_pkg.__version__
    assert isinstance(v, str), type(v)
    assert _SEMVER_RE.match(v), (
        f"__version__={v!r} is not a plain X.Y.Z SemVer triple. "
        f"Either revert whatever introduced the non-standard string, "
        f"or update this test to match the new convention."
    )
