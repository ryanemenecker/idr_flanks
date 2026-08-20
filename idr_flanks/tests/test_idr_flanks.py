"""
Unit and regression test for the idr_flanks package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import idr_flanks


def test_idr_flanks_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "idr_flanks" in sys.modules
