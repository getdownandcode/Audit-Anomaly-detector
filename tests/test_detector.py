import numpy as np
import pandas as pd

from detector import benford_test, prepare, risk_score, rule_checks


def sample() -> pd.DataFrame:
    return pd.DataFrame({"Time": [0, 60, 120, 7200], "Amount": [100, 100, 100, 250], "V1": [0.0, 0.1, 0.2, 4.0]})


def test_prepare_drops_invalid_amounts_and_preserves_index() -> None:
    data = sample().assign(Amount=[100, 0, -1, 250])
    result = prepare(data)
    assert result.index.tolist() == [0, 3]
    assert result["txn_id"].tolist() == [0, 3]


def test_benford_frequencies_and_leading_digits() -> None:
    data = prepare(sample())
    expected, _, _ = benford_test(data)
    assert np.isclose(expected.sum(), 1)
    assert data["leading_digit"].tolist() == [1, 1, 1, 2]


def test_repeat_amount_within_one_hour_is_detected() -> None:
    result = rule_checks(prepare(sample()))
    assert result["repeat_amount"].tolist() == [True, True, True, False]


def test_risk_is_bounded_and_reasons_are_readable() -> None:
    result = risk_score(prepare(sample()))
    assert result["risk"].between(0, 1).all()
    assert result["reasons"].ne("").all()
