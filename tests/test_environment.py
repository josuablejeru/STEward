import platform

import steward
from steward.environment import collect_environment_info


def test_environment_info_fields() -> None:
    info = collect_environment_info()
    assert info.steward_version == steward.__version__
    assert info.python_version == platform.python_version()
    assert info.ste_reference == "ASD-STE100 Issue 9 (January 2025)"


def test_environment_info_values_are_strings() -> None:
    data = collect_environment_info().to_dict()
    assert data
    assert all(isinstance(v, str) for v in data.values())


def test_runs_on_python_312_or_newer() -> None:
    assert tuple(int(p) for p in platform.python_version_tuple()[:2]) >= (3, 12)
