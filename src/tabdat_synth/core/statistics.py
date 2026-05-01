from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import polars as pl

from tabdat_synth.domain.contracts import CoefficientSumSpec


def create_one_hot_name(
  feature_name: str,
  value: str,
  reference_value: str = "99",
) -> str | None:
  """Create one-hot encoded feature name unless value is the reference."""
  return None if value == reference_value else f"{feature_name}_c_{value}"


def calculate_linear_predictor(
  row: Mapping[str, object],
  coefficients: Mapping[str, float],
  reference_value: str = "99",
) -> float:
  """Calculate a row-level linear predictor from one-hot coefficient names."""
  score = coefficients["intercept"]

  for col_name, col_value in row.items():
    if col_name in ["age_cont"]:
      continue

    feature_name = create_one_hot_name(col_name, str(col_value), reference_value)
    if feature_name and feature_name in coefficients:
      score += coefficients[feature_name]

  return score


def calculate_weighted_sum(
  row: Mapping[str, object],
  coefficient_sum: CoefficientSumSpec,
) -> float:
  """Calculate a coefficient-based weighted sum for one generated row."""
  score = coefficient_sum.intercept

  for term in coefficient_sum.terms:
    value = row[term.feature_name]
    if term.value is None:
      score += term.coefficient * float(cast(Any, value))
      continue

    if str(value) == term.value:
      score += term.coefficient

  return score


def calculate_weighted_sums(
  df: pl.DataFrame,
  coefficient_sum: CoefficientSumSpec,
) -> np.ndarray:
  """Calculate coefficient-based weighted sums for a dataframe."""
  if df.height == 0:
    return np.array([], dtype=float)
  if not coefficient_sum.terms:
    return np.full(df.height, coefficient_sum.intercept, dtype=float)

  score = pl.lit(coefficient_sum.intercept, dtype=pl.Float64)
  for term in coefficient_sum.terms:
    coefficient = pl.lit(term.coefficient, dtype=pl.Float64)
    if term.value is None:
      term_expr = pl.col(term.feature_name).cast(pl.Float64, strict=False)
    else:
      term_expr = (
        pl.col(term.feature_name)
        .cast(pl.String)
        .eq(pl.lit(term.value))
        .fill_null(False)
        .cast(pl.Float64)
      )
    score += term_expr * coefficient

  return df.select(score.alias("weighted_sum"))["weighted_sum"].to_numpy()


def compute_category_frequencies(series: pl.Series) -> dict[str, float]:
  """Compute observed category proportions for a Polars series."""
  counts = series.value_counts().sort("count", descending=True)
  total = counts["count"].sum()
  return {str(row[0]): row[1] / total for row in counts.iter_rows()}
