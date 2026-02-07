import pytest

from services.auth_service import validate_password_strength
from services.lead_service import lead_temperature


@pytest.mark.parametrize(
    ("probabilidade", "score", "expected"),
    [
        (None, None, "unknown"),
        (0.7, None, "hot"),
        (0.349, None, "cold"),
        (0.35, None, "warm"),
        (0.699, None, "warm"),
        (None, 70, "hot"),
        (None, 35, "warm"),
        (None, 34, "cold"),
    ],
)
def test_lead_temperature_thresholds(probabilidade, score, expected):
    assert lead_temperature(probabilidade, score) == expected


@pytest.mark.parametrize(
    ("password", "expected_ok"),
    [
        ("Senha123!", True),
        ("Senha123", False),
        ("senha123!", False),
        ("SENHA123!", False),
        ("Senha!!", False),
    ],
)
def test_validate_password_strength(password, expected_ok):
    ok, _ = validate_password_strength(password)
    assert ok is expected_ok
