"""Safety audit regression: no real inputs, authorization, or diagnostic execution."""

import ast
import inspect

import pytest

from src.research import v10_economic_execution as execution


def test_alternate_root_cannot_reach_sample_loading_without_authorization(
    tmp_path, monkeypatch
):
    def forbidden_loader(*args, **kwargs):
        pytest.fail("unguarded executor reached sample loading without authorization")

    monkeypatch.setattr(execution, "_merged_records", forbidden_loader)
    monkeypatch.setattr(execution, "sample_session", forbidden_loader)
    monkeypatch.setattr(execution, "build_diagnostic", forbidden_loader)
    # Retain the original attack: the old callable must not survive as an alias.
    with pytest.raises(AttributeError):
        execution.reserved_execution(
            tmp_path / "synthetic_alternate_root",
            {"synthetic": True},
            integrity_check=lambda: None,
            load_samples=forbidden_loader,
        )
    # The sole remaining API cannot accept the same injected roots/callbacks.
    for kwargs in (
        {"directory": tmp_path / "alternate"},
        {"output_root": tmp_path / "alternate"},
        {"integrity_check": lambda: None},
        {"load_samples": forbidden_loader},
        {"manifest": {}},
        {"data_root": tmp_path},
        {"permutations": 1},
        {"executor_identity": {}},
        {"diagnostic": forbidden_loader},
    ):
        with pytest.raises(TypeError):
            execution.execute_once(**kwargs)
    monkeypatch.setattr(execution, "WORKSPACE", tmp_path)
    monkeypatch.setattr(execution, "AUTHORIZATION", tmp_path / "absent.json")
    monkeypatch.setattr(execution, "REPORT_DIRECTORY", tmp_path / "reserved")
    with pytest.raises(ValueError, match="authorization is absent"):
        execution.execute_once()
    assert not execution.REPORT_DIRECTORY.exists()


def test_only_execute_once_owns_loading_diagnostic_and_publication():
    tree = ast.parse(inspect.getsource(execution))
    assert not inspect.signature(execution.execute_once).parameters
    # Nested closures are owned by execute_once, not independently callable entry points.
    owners = {}
    for function in tree.body:
        if isinstance(function, ast.FunctionDef):
            owners[function.name] = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
    for operation in (
        "sample_session",
        "_merged_records",
        "build_diagnostic",
        "asdict",
    ):
        assert [name for name, calls in owners.items() if operation in calls] == [
            "execute_once"
        ]
    assert owners["reserve"].isdisjoint(
        {"sample_session", "build_diagnostic", "execute_once"}
    )
