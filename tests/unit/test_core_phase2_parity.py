from __future__ import annotations

import pytest

from tabdat_synth.core.probability import derive_child_seed, logit, sigmoid
from tabdat_synth.core.sampling import (
  apply_freq_shifts,
  generate_responses,
)
from tabdat_synth.core.statistics import (
  calculate_linear_predictor,
  calculate_weighted_sum,
  calculate_weighted_sums,
  compute_category_frequencies,
)
from tabdat_synth.domain.contracts import (
  CoefficientSumSpec,
  CoefficientTerm,
  FreqShift,
)


def test_probability_helpers_and_child_seed_match_legacy_behavior():
  import numpy as np

  result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))

  assert np.isfinite(result).all()
  assert result[0] < 1.0e-100
  assert result[1] == pytest.approx(0.5)
  assert result[2] == pytest.approx(1.0)

  logits = logit(np.array([0.0, 0.5, 1.0]), epsilon=1.0e-6)
  assert np.isfinite(logits).all()
  assert logits[1] == pytest.approx(0.0)

  assert derive_child_seed(343, 10_001) == derive_child_seed(343, 10_001)
  assert derive_child_seed(343, 10_001) != derive_child_seed(343, 10_002)


def test_frequency_shifts_use_callback_for_missing_category_messages():
  messages = []

  result = apply_freq_shifts(
    {"0": 0.25, "1": 0.75},
    (FreqShift(source="9", dest="1", shift_pct=10.0),),
    messages.append,
  )

  assert result == pytest.approx({"0": 0.25, "1": 0.75})
  assert messages == ["source category '9' not found in distribution"]


def test_frequency_shifts_reject_negative_frequencies():
  with pytest.raises(ValueError, match="negative frequencies"):
    apply_freq_shifts(
      {"0": 0.10, "1": 0.90},
      (FreqShift(source="0", dest="1", shift_pct=20.0),),
    )


def test_category_frequency_and_weighted_sum_core_behavior():
  import polars as pl

  series = pl.Series("priority", ["1", "0", "1", "99"])
  assert compute_category_frequencies(series) == pytest.approx(
    {"1": 0.50, "0": 0.25, "99": 0.25}
  )

  coefficients = {
    "intercept": 1.25,
    "Albumin_c_1": 0.75,
    "Prior1245_c_2": -0.5,
  }
  row = {"age_cont": 92, "agegrp": "99", "Albumin": "1", "Prior1245": "2"}

  assert calculate_linear_predictor(row, coefficients) == pytest.approx(1.5)

  coefficient_sum = CoefficientSumSpec(
    intercept=0.5,
    terms=(
      CoefficientTerm(feature_name="Albumin", coefficient=0.75, value="1"),
      CoefficientTerm(feature_name="score", coefficient=0.25),
    ),
  )
  weighted_row = {"Albumin": "1", "score": 4.0}
  assert calculate_weighted_sum(weighted_row, coefficient_sum) == pytest.approx(2.25)

  frame = pl.DataFrame({"Albumin": ["1", "0"], "score": [4.0, 8.0]})
  weighted_sums = calculate_weighted_sums(frame, coefficient_sum)
  assert weighted_sums.tolist() == pytest.approx([2.25, 2.5])


def test_weighted_sums_vectorized_terms_match_row_calculation():
  import polars as pl

  coefficient_sum = CoefficientSumSpec(
    intercept=-0.5,
    terms=(
      CoefficientTerm(feature_name="group", coefficient=1.25, value="case"),
      CoefficientTerm(feature_name="score", coefficient=0.5),
      CoefficientTerm(feature_name="flag", coefficient=-0.75),
    ),
  )
  frame = pl.DataFrame(
    {
      "group": ["case", "control", "case"],
      "score": [2, 4, 6],
      "flag": [1, 0, 1],
    }
  )

  row_values = [
    calculate_weighted_sum(row, coefficient_sum) for row in frame.to_dicts()
  ]
  weighted_sums = calculate_weighted_sums(frame, coefficient_sum)

  assert weighted_sums.tolist() == pytest.approx(row_values)


def test_weighted_sums_vectorized_intercept_only_and_empty_frames():
  import polars as pl

  coefficient_sum = CoefficientSumSpec(intercept=2.75)

  intercept_only = calculate_weighted_sums(
    pl.DataFrame({"unused": ["a", "b", "c"]}),
    coefficient_sum,
  )
  empty = calculate_weighted_sums(pl.DataFrame({"unused": []}), coefficient_sum)

  assert intercept_only.tolist() == pytest.approx([2.75, 2.75, 2.75])
  assert empty.tolist() == []


def test_response_generation_is_seed_deterministic():
  import numpy as np

  log_odds = np.array([-1.0, 0.0, 1.0])

  risk_a, outcomes_a = generate_responses(
    log_odds, intercept=0.0, slope=1.0, seed=2026
  )
  risk_b, outcomes_b = generate_responses(
    log_odds, intercept=0.0, slope=1.0, seed=2026
  )

  assert np.allclose(risk_a, sigmoid(log_odds))
  assert np.allclose(risk_a, risk_b)
  assert np.array_equal(outcomes_a, outcomes_b)
  assert set(outcomes_a.tolist()) <= {0, 1}
