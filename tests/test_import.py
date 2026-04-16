"""Test probLM-solver."""

import problm_solver


def test_import() -> None:
    """Test that the app can be imported."""
    assert isinstance(problm_solver.__name__, str)
