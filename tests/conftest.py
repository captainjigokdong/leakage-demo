import pytest

from synth.generate import generate


@pytest.fixture(scope="session")
def small_tables():
    """환자 300명 합성 데이터 (빠른 시험용)."""
    return generate(seed=7, n_patients=300).tables
