"""Golden test for DFM: today's API must reproduce the captured fixtures."""

import pytest

from ._cases import cases_for, check_case


@pytest.mark.parametrize("case", cases_for("dfm"), ids=repr)
def test_golden_dfm(case, pinned):
    check_case(case.name, case.run())
