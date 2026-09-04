"""The API module must be importable in an environment without fastapi.

Importing a module should never have the side effect of building an
application. These tests pin that: the module imports, and the failure only
arrives when someone actually asks for the app.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

import cke.api
import cke.api.server as server


def test_module_imports_regardless_of_fastapi():
    """Importing the module must not build the app."""
    module = importlib.reload(server)
    assert module.create_app is not None


def test_app_is_not_built_at_import_time():
    """No module-scope `app` binding; it is produced on attribute access.

    This used to read the module's own source through its __file__ and assert
    that the string "\napp = create_app()" did not appear in it. That is an
    assertion about a spelling: it passes on `app=create_app()` without the
    space, on the binding being built in any other way, and on a comment
    mentioning it. Import the module and look at what is actually bound.
    """
    module = importlib.reload(server)

    assert "app" not in vars(module), (
        "the app was built at import time; it must be produced by "
        "__getattr__ on first access so the module imports without fastapi"
    )

    built = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(module, "get_app", lambda: built.append("app") or "an app")
        assert module.app == "an app"
    finally:
        monkeypatch.undo()

    assert built == ["app"], "attribute access did not build the app"


@pytest.mark.parametrize("module", [server, cke.api])
def test_app_attribute_raises_clearly_without_fastapi(module, monkeypatch):
    """Accessing .app without fastapi names the dependency and how to get it."""
    monkeypatch.setattr(server, "FastAPI", None)

    with pytest.raises(server.MissingDependencyError) as excinfo:
        getattr(module, "app")

    message = str(excinfo.value)
    assert "fastapi" in message
    assert "pip install fastapi" in message


def test_create_app_raises_without_fastapi(monkeypatch):
    monkeypatch.setattr(server, "FastAPI", None)

    with pytest.raises(server.MissingDependencyError):
        server.create_app()


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        server.no_such_attribute  # noqa: B018

    with pytest.raises(AttributeError):
        cke.api.no_such_attribute  # noqa: B018


def test_import_survives_fastapi_being_absent(monkeypatch):
    """Simulate a bare environment and re-import from scratch."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "fastapi" or name.startswith("fastapi."):
            raise ImportError("No module named 'fastapi'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    module = importlib.reload(server)

    assert module.FastAPI is None
    with pytest.raises(module.MissingDependencyError):
        module.create_app()


def test_the_app_is_built_once_and_cached(monkeypatch):
    """Repeated access must return the same object, as the eager module did.

    Routes, middleware or state registered on one access have to still be
    there on the next.
    """

    class FakeFastAPI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get(self, path):
            return lambda fn: fn

        def post(self, path):
            return lambda fn: fn

    monkeypatch.setattr(server, "FastAPI", FakeFastAPI)
    server.reset_app()
    try:
        assert server.app is server.app
        assert cke.api.app is server.app
    finally:
        server.reset_app()


def test_reset_app_forces_a_rebuild(monkeypatch):
    class FakeFastAPI:
        def __init__(self, **kwargs):
            pass

        def get(self, path):
            return lambda fn: fn

        def post(self, path):
            return lambda fn: fn

    monkeypatch.setattr(server, "FastAPI", FakeFastAPI)
    server.reset_app()
    try:
        first = server.app
        server.reset_app()
        assert server.app is not first
    finally:
        server.reset_app()


def test_star_import_does_not_build_the_app():
    """ "app" in __all__ made `from cke.api import *` build the application,
    which defeats the point of the module being importable without fastapi."""
    assert "app" not in cke.api.__all__

    namespace: dict = {}
    exec("from cke.api import *", namespace)  # noqa: S102

    assert "create_app" in namespace
    assert "app" not in namespace
