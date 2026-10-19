"""Golden test for DFM.

``test_golden_dfm_engine`` runs the engine (the pre-redesign class, now
``DFMEngine``) exactly as the fixtures were captured, update path included
(the fixtures were captured with a test-only import shim around ``update``,
whose broken import the engine move fixed), and must reproduce every output
bitwise on the capture platform.
"""

import pytest

from ._cases import cases_for, check_case


@pytest.mark.parametrize("case", cases_for("dfm"), ids=repr)
def test_golden_dfm_engine(case, pinned):
    check_case(case.name, case.run())
