"""Shared pytest configuration for the seals suite.

Devstack-standard two-tier split (see the EE Spec "Testing across the stack",
mirroring gtappy_tests/conftest.py): most tests construct their own data and run
anywhere; tests marked ``requires_base_data`` need the local ``base_data`` store
and SKIP cleanly when it is absent, and ``slow`` marks minutes-not-seconds tests.

The automated tier (what a CI workflow runs) is::

    pytest -m "not requires_base_data and not slow"
"""

import os

import pytest

# Matches the base_data convention used across the devstack.
BASE_DATA_DIR = os.path.join(os.path.expanduser('~'), 'Files', 'base_data')


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_base_data: needs the shared base_data tree; skipped when absent",
    )
    config.addinivalue_line(
        "markers",
        "slow: minutes rather than seconds even with data present; excluded from the automated tier",
    )


def pytest_collection_modifyitems(config, items):
    if os.path.isdir(BASE_DATA_DIR):
        return

    skip_marker = pytest.mark.skip(
        reason="requires base_data; expected %s" % BASE_DATA_DIR)
    for item in items:
        if "requires_base_data" in item.keywords:
            item.add_marker(skip_marker)
