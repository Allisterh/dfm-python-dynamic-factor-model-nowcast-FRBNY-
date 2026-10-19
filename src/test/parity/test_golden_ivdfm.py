"""Golden test for iVDFM: today's API must reproduce the captured fixtures."""

import pytest

from ._cases import cases_for, check_case


@pytest.mark.parametrize("case", cases_for("ivdfm"), ids=repr)
def test_golden_ivdfm(case, pinned):
    check_case(case.name, case.run())
