from __future__ import annotations

import polars as pl
import pytest

from tabdat_synth.core.evaluation import (
  categorical_total_variation_distance,
  duplicate_row_rate,
  exact_row_match_rate,
  nearest_neighbor_distance_summary,
  null_rate_delta,
  numeric_ks_statistic,
  numeric_summary_deltas,
  unseen_synthetic_category_rate,
)


def test_categorical_metrics_compare_frequencies_and_unseen_values():
  source = pl.Series("category", ["a", "a", "b", None])
  synthetic = pl.Series("category", ["a", "b", "c", "c"])

  assert categorical_total_variation_distance(source, synthetic) == pytest.approx(0.5)
  assert unseen_synthetic_category_rate(source, synthetic) == pytest.approx(0.5)
  assert null_rate_delta(source, synthetic) == pytest.approx(-0.25)


def test_numeric_metrics_report_summary_deltas_and_ks_statistic():
  source = pl.Series("score", [1.0, 2.0, 3.0])
  synthetic = pl.Series("score", [2.0, 4.0, 6.0])

  summary = numeric_summary_deltas(source, synthetic)

  assert summary["source_mean"] == pytest.approx(2.0)
  assert summary["synthetic_mean"] == pytest.approx(4.0)
  assert summary["mean_delta"] == pytest.approx(2.0)
  assert summary["std_delta"] == pytest.approx(0.81649658)
  assert summary["min_delta"] == pytest.approx(1.0)
  assert summary["max_delta"] == pytest.approx(3.0)
  assert numeric_ks_statistic(source, synthetic) == pytest.approx(2 / 3)


def test_row_risk_metrics_report_exact_matches_and_duplicates():
  source = pl.DataFrame({"category": ["a", "b"], "score": [1, 2]})
  synthetic = pl.DataFrame({"category": ["a", "a", "c"], "score": [1, 1, 3]})

  assert exact_row_match_rate(source, synthetic) == pytest.approx(2 / 3)
  assert duplicate_row_rate(synthetic) == pytest.approx(1 / 3)


def test_nearest_neighbor_distance_summary_uses_scaled_numeric_distance():
  source = pl.DataFrame({"score": [0.0, 2.0]})
  synthetic = pl.DataFrame({"score": [1.0]})

  summary = nearest_neighbor_distance_summary(
    source,
    synthetic,
    ("score",),
    max_pairwise_rows=10,
    random_seed=123,
  )

  assert summary["source_rows_compared"] == 2
  assert summary["synthetic_rows_compared"] == 1
  assert summary["column_count"] == 1
  assert summary["mean"] == pytest.approx(1.0)
  assert summary["min"] == pytest.approx(1.0)
  assert summary["max"] == pytest.approx(1.0)


def test_nearest_neighbor_distance_summary_requires_positive_cap():
  with pytest.raises(ValueError, match="max_pairwise_rows must be positive"):
    nearest_neighbor_distance_summary(
      pl.DataFrame({"score": [1]}),
      pl.DataFrame({"score": [1]}),
      ("score",),
      max_pairwise_rows=0,
    )
