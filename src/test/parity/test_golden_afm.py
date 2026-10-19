"""Golden test for AFM: today's API must reproduce the captured fixtures."""

import pytest

from ._cases import cases_for, check_case


@pytest.mark.parametrize("case", cases_for("afm"), ids=repr)
def test_golden_afm(case, pinned):
    check_case(case.name, case.run())
