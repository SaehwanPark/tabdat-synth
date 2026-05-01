from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl
from scipy.stats import ks_2samp


def null_rate(series: pl.Series) -> float:
  """Return the fraction of null rows in a series."""
  if len(series) == 0:
    return 0.0
  return series.null_count() / len(series)


def null_rate_delta(source: pl.Series, synthetic: pl.Series) -> float:
  """Return synthetic null-rate minus source null-rate."""
  return null_rate(synthetic) - null_rate(source)


def categorical_total_variation_distance(
  source: pl.Series,
  synthetic: pl.Series,
) -> float:
  """Compare categorical distributions with total-variation distance."""
  source_values = _string_values(source)
  synthetic_values = _string_values(synthetic)
  if source_values.size == 0 and synthetic_values.size == 0:
    return 0.0
  if source_values.size == 0 or synthetic_values.size == 0:
    return 1.0

  source_freq = _category_proportions(source_values)
  synthetic_freq = _category_proportions(synthetic_values)
  categories = set(source_freq) | set(synthetic_freq)
  distance = sum(
    abs(synthetic_freq.get(category, 0.0) - source_freq.get(category, 0.0))
    for category in categories
  )
  return 0.5 * distance


def unseen_synthetic_category_rate(
  source: pl.Series,
  synthetic: pl.Series,
) -> float:
  """Return the share of synthetic non-null values unseen in source data."""
  source_values = _string_values(source)
  synthetic_values = _string_values(synthetic)
  if synthetic_values.size == 0:
    return 0.0
  if source_values.size == 0:
    return 1.0

  observed = set(source_values.tolist())
  unseen_count = sum(value not in observed for value in synthetic_values.tolist())
  return unseen_count / synthetic_values.size


def numeric_summary_deltas(
  source: pl.Series,
  synthetic: pl.Series,
) -> dict[str, float | None]:
  """Return numeric summary statistics and synthetic-minus-source deltas."""
  source_summary = _numeric_summary(_finite_numeric_values(source))
  synthetic_summary = _numeric_summary(_finite_numeric_values(synthetic))
  return {
    "source_mean": source_summary["mean"],
    "synthetic_mean": synthetic_summary["mean"],
    "mean_delta": _optional_delta(
      source_summary["mean"], synthetic_summary["mean"]
    ),
    "source_std": source_summary["std"],
    "synthetic_std": synthetic_summary["std"],
    "std_delta": _optional_delta(source_summary["std"], synthetic_summary["std"]),
    "source_min": source_summary["min"],
    "synthetic_min": synthetic_summary["min"],
    "min_delta": _optional_delta(source_summary["min"], synthetic_summary["min"]),
    "source_max": source_summary["max"],
    "synthetic_max": synthetic_summary["max"],
    "max_delta": _optional_delta(source_summary["max"], synthetic_summary["max"]),
  }


def numeric_ks_statistic(source: pl.Series, synthetic: pl.Series) -> float | None:
  """Return a two-sample Kolmogorov-Smirnov statistic for numeric values."""
  source_values = _finite_numeric_values(source)
  synthetic_values = _finite_numeric_values(synthetic)
  if source_values.size == 0 or synthetic_values.size == 0:
    return None
  return float(ks_2samp(source_values, synthetic_values).statistic)


def exact_row_match_rate(source_df: pl.DataFrame, synthetic_df: pl.DataFrame) -> float:
  """Return the fraction of synthetic rows exactly present in the source data."""
  if synthetic_df.height == 0:
    return 0.0
  _validate_matching_columns(source_df, synthetic_df)
  if source_df.height == 0 or source_df.width == 0:
    return 0.0

  source_rows = set(source_df.iter_rows())
  matched = sum(row in source_rows for row in synthetic_df.iter_rows())
  return matched / synthetic_df.height


def duplicate_row_rate(df: pl.DataFrame) -> float:
  """Return the fraction of rows beyond the unique-row count."""
  if df.height == 0:
    return 0.0
  return (df.height - df.unique().height) / df.height


def nearest_neighbor_distance_summary(
  source_df: pl.DataFrame,
  synthetic_df: pl.DataFrame,
  columns: Iterable[str],
  *,
  max_pairwise_rows: int = 1000,
  random_seed: int = 343,
) -> dict[str, float | int | None]:
  """Return synthetic-to-source nearest-neighbor distances over numeric columns."""
  if max_pairwise_rows <= 0:
    raise ValueError("max_pairwise_rows must be positive")

  resolved_columns = tuple(columns)
  source_matrix = _numeric_matrix(source_df, resolved_columns)
  synthetic_matrix = _numeric_matrix(synthetic_df, resolved_columns)
  if (
    not resolved_columns
    or source_matrix.shape[0] == 0
    or synthetic_matrix.shape[0] == 0
  ):
    return _empty_nearest_neighbor_summary(
      source_matrix.shape[0],
      synthetic_matrix.shape[0],
      len(resolved_columns),
    )

  rng = np.random.default_rng(random_seed)
  source_sample = _sample_rows(source_matrix, max_pairwise_rows, rng)
  synthetic_sample = _sample_rows(synthetic_matrix, max_pairwise_rows, rng)
  scaled_source, scaled_synthetic = _scale_against_source(
    source_sample,
    synthetic_sample,
  )
  nearest = _nearest_distances(scaled_source, scaled_synthetic)

  return {
    "source_rows_compared": int(source_sample.shape[0]),
    "synthetic_rows_compared": int(synthetic_sample.shape[0]),
    "column_count": len(resolved_columns),
    "min": float(np.min(nearest)),
    "p50": float(np.percentile(nearest, 50)),
    "p95": float(np.percentile(nearest, 95)),
    "mean": float(np.mean(nearest)),
    "max": float(np.max(nearest)),
  }


def _string_values(series: pl.Series) -> np.ndarray:
  return series.cast(pl.String, strict=False).drop_nulls().to_numpy()


def _category_proportions(values: np.ndarray) -> dict[str, float]:
  unique_values, counts = np.unique(values.astype(str), return_counts=True)
  total = float(values.size)
  return {
    str(value): float(count) / total
    for value, count in zip(unique_values, counts, strict=True)
  }


def _finite_numeric_values(series: pl.Series) -> np.ndarray:
  values = series.cast(pl.Float64, strict=False).drop_nulls().to_numpy()
  numeric = np.asarray(values, dtype=float)
  return numeric[np.isfinite(numeric)]


def _numeric_summary(values: np.ndarray) -> dict[str, float | None]:
  if values.size == 0:
    return {"mean": None, "std": None, "min": None, "max": None}
  return {
    "mean": float(np.mean(values)),
    "std": float(np.std(values)),
    "min": float(np.min(values)),
    "max": float(np.max(values)),
  }


def _optional_delta(
  source_value: float | None,
  synthetic_value: float | None,
) -> float | None:
  if source_value is None or synthetic_value is None:
    return None
  return synthetic_value - source_value


def _validate_matching_columns(
  source_df: pl.DataFrame,
  synthetic_df: pl.DataFrame,
) -> None:
  if source_df.columns != synthetic_df.columns:
    raise ValueError("source_df and synthetic_df must have matching columns")


def _numeric_matrix(df: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
  if not columns or df.height == 0:
    return np.empty((0, len(columns)), dtype=float)
  numeric_df = df.select(
    pl.col(column).cast(pl.Float64, strict=False).alias(column)
    for column in columns
  )
  matrix = np.asarray(numeric_df.to_numpy(), dtype=float)
  if matrix.size == 0:
    return np.empty((0, len(columns)), dtype=float)
  finite_rows = np.isfinite(matrix).all(axis=1)
  return matrix[finite_rows]


def _empty_nearest_neighbor_summary(
  source_rows: int,
  synthetic_rows: int,
  column_count: int,
) -> dict[str, float | int | None]:
  return {
    "source_rows_compared": source_rows,
    "synthetic_rows_compared": synthetic_rows,
    "column_count": column_count,
    "min": None,
    "p50": None,
    "p95": None,
    "mean": None,
    "max": None,
  }


def _sample_rows(
  matrix: np.ndarray,
  max_rows: int,
  rng: np.random.Generator,
) -> np.ndarray:
  if matrix.shape[0] <= max_rows:
    return matrix
  indices = rng.choice(matrix.shape[0], size=max_rows, replace=False)
  return matrix[np.sort(indices)]


def _scale_against_source(
  source_matrix: np.ndarray,
  synthetic_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  center = np.mean(source_matrix, axis=0)
  scale = np.std(source_matrix, axis=0)
  scale = np.where(scale > 0, scale, 1.0)
  return (source_matrix - center) / scale, (synthetic_matrix - center) / scale


def _nearest_distances(
  source_matrix: np.ndarray,
  synthetic_matrix: np.ndarray,
) -> np.ndarray:
  distances = np.empty(synthetic_matrix.shape[0], dtype=float)
  for index, synthetic_row in enumerate(synthetic_matrix):
    row_distances = np.linalg.norm(source_matrix - synthetic_row, axis=1)
    distances[index] = float(np.min(row_distances))
  return distances
