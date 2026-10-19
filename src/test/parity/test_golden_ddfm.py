"""Golden tests for DDFM.

``test_golden_ddfm_engine`` runs the engine (the pre-redesign class, now
``DDFMEngine``) exactly as the fixtures were captured.
"""

import pytest

from ._cases import cases_for, check_case


@pytest.mark.parametrize("case", cases_for("ddfm"), ids=repr)
def test_golden_ddfm_engine(case, pinned):
    check_case(case.name, case.run())
