"""Smoke test for the Streamlit dashboard (app.py).

Uses streamlit.testing.v1.AppTest to verify the app starts without exceptions.
The test does NOT click the "Run CFR Solver" button (that would trigger a ~70s
solve); it verifies the initial render (DP heat maps, Plotly lookup, bankroll
MC simulation) completes within the timeout budget.
"""

import pytest

try:
    from streamlit.testing.v1 import AppTest

    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="streamlit not installed")
def test_app_runs_without_exception():
    """App renders all four tabs without raising an exception."""
    at = AppTest.from_file("app.py")
    at.run(timeout=120)
    assert not at.exception, f"App raised an exception: {at.exception}"


@pytest.mark.skipif(not _STREAMLIT_AVAILABLE, reason="streamlit not installed")
def test_app_has_expected_tabs():
    """App exposes the four expected tab labels."""
    at = AppTest.from_file("app.py")
    at.run(timeout=120)
    tab_labels = [t.label for t in at.tabs]
    assert "Strategy Heat Maps" in tab_labels
    assert "Interactive Plotly Lookup" in tab_labels
    assert "Bankroll Analysis" in tab_labels
    assert "Strategy Report" in tab_labels
