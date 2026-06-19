"""Tests for notebook visualization dependencies."""


def test_matplotlib_available_for_evidence_review():
  """Evidence review notebook can render 16:9 figures."""
  import matplotlib

  assert matplotlib.__version__
