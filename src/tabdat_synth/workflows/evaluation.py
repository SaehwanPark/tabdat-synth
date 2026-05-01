from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from tabdat_synth.core.evaluation import (
  categorical_total_variation_distance,
  duplicate_row_rate,
  exact_row_match_rate,
  nearest_neighbor_distance_summary,
  null_rate,
  null_rate_delta,
  numeric_ks_statistic,
  numeric_summary_deltas,
  unseen_synthetic_category_rate,
)
from tabdat_synth.domain.contracts import (
  DatasetSchema,
  EvaluationReport,
  GenerationSpec,
  MetricResult,
  MetricScalar,
)
from tabdat_synth.interfaces.io import (
  load_source_dataframe,
  prepare_source_dataframe,
)


def evaluate_synthetic_data(
  source_df: pl.DataFrame,
  synthetic_df: pl.DataFrame,
  schema: DatasetSchema,
  *,
  identifier_columns: Iterable[str] = (),
  nearest_neighbor_columns: Iterable[str] | None = None,
  max_pairwise_rows: int = 1000,
  random_seed: int = 343,
) -> EvaluationReport:
  """Evaluate synthetic data quality and practical disclosure-risk heuristics."""
  if max_pairwise_rows <= 0:
    raise ValueError("max_pairwise_rows must be positive")

  identifier_set = _identifier_columns(schema, identifier_columns)
  evaluated_columns = _evaluated_columns(source_df, synthetic_df, schema, identifier_set)
  skipped_columns = _skipped_columns(source_df, synthetic_df, schema, identifier_set)

  quality_metrics = _quality_metrics(
    source_df,
    synthetic_df,
    schema,
    evaluated_columns,
  )
  disclosure_metrics = _disclosure_risk_metrics(
    source_df,
    synthetic_df,
    schema,
    evaluated_columns,
    nearest_neighbor_columns=nearest_neighbor_columns,
    max_pairwise_rows=max_pairwise_rows,
    random_seed=random_seed,
  )

  return EvaluationReport(
    source_row_count=source_df.height,
    synthetic_row_count=synthetic_df.height,
    evaluated_columns=evaluated_columns,
    skipped_columns=skipped_columns,
    quality_metrics=tuple(quality_metrics),
    disclosure_risk_metrics=tuple(disclosure_metrics),
  )


def evaluate_from_spec(
  spec: GenerationSpec,
  synthetic_df: pl.DataFrame,
  *,
  source_df: pl.DataFrame | None = None,
  identifier_columns: Iterable[str] = (),
  nearest_neighbor_columns: Iterable[str] | None = None,
  max_pairwise_rows: int = 1000,
) -> EvaluationReport:
  """Evaluate generated rows against the source dataset declared by a spec."""
  raw_source_df = (
    source_df
    if source_df is not None
    else load_source_dataframe(
      spec.source_path,
      spec.source_format,
      member=spec.source_member,
    )
  )
  source_columns = tuple(
    column.name for column in spec.schema.columns if column.name in raw_source_df.columns
  )
  prepared_source_df = prepare_source_dataframe(
    raw_source_df,
    spec.schema,
    columns=source_columns,
  )
  return evaluate_synthetic_data(
    prepared_source_df,
    synthetic_df,
    spec.schema,
    identifier_columns=identifier_columns,
    nearest_neighbor_columns=nearest_neighbor_columns,
    max_pairwise_rows=max_pairwise_rows,
    random_seed=spec.random_seed,
  )


def _identifier_columns(
  schema: DatasetSchema,
  explicit_identifier_columns: Iterable[str],
) -> set[str]:
  return {
    column.name for column in schema.columns if column.role == "identifier"
  } | set(explicit_identifier_columns)


def _evaluated_columns(
  source_df: pl.DataFrame,
  synthetic_df: pl.DataFrame,
  schema: DatasetSchema,
  identifier_columns: set[str],
) -> tuple[str, ...]:
  return tuple(
    column.name
    for column in schema.columns
    if column.name not in identifier_columns
    and column.name in source_df.columns
    and column.name in synthetic_df.columns
  )


def _skipped_columns(
  source_df: pl.DataFrame,
  synthetic_df: pl.DataFrame,
  schema: DatasetSchema,
  identifier_columns: set[str],
) -> tuple[str, ...]:
  return tuple(
    column.name
    for column in schema.columns
    if column.name not in identifier_columns
    and (column.name not in source_df.columns or column.name not in synthetic_df.columns)
  )


def _quality_metrics(
  source_df: pl.DataFrame,
  synthetic_df: pl.DataFrame,
  schema: DatasetSchema,
  evaluated_columns: tuple[str, ...],
) -> list[MetricResult]:
  schema_columns = schema.columns_by_name
  metrics: list[MetricResult] = []

  for column_name in evaluated_columns:
    column_spec = schema_columns[column_name]
    source_series = source_df[column_name]
    synthetic_series = synthetic_df[column_name]
    metrics.append(
      MetricResult(
        category="quality",
        name="null_rate_delta",
        column=column_name,
        values={
          "source_null_rate": null_rate(source_series),
          "synthetic_null_rate": null_rate(synthetic_series),
          "delta": null_rate_delta(source_series, synthetic_series),
        },
      )
    )

    if column_spec.data_type == "categorical":
      metrics.extend(
        _categorical_quality_metrics(column_name, source_series, synthetic_series)
      )
    else:
      metrics.extend(
        _numeric_quality_metrics(column_name, source_series, synthetic_series)
      )

  return metrics


def _categorical_quality_metrics(
  column_name: str,
  source_series: pl.Series,
  synthetic_series: pl.Series,
) -> list[MetricResult]:
  return [
    MetricResult(
      category="quality",
      name="categorical_total_variation_distance",
      column=column_name,
      values={
        "value": categorical_total_variation_distance(
          source_series,
          synthetic_series,
        )
      },
    ),
    MetricResult(
      category="quality",
      name="unseen_synthetic_category_rate",
      column=column_name,
      values={"value": unseen_synthetic_category_rate(source_series, synthetic_series)},
    ),
  ]


def _numeric_quality_metrics(
  column_name: str,
  source_series: pl.Series,
  synthetic_series: pl.Series,
) -> list[MetricResult]:
  return [
    MetricResult(
      category="quality",
      name="numeric_summary_delta",
      column=column_name,
      values=numeric_summary_deltas(source_series, synthetic_series),
    ),
    MetricResult(
      category="quality",
      name="numeric_ks_statistic",
      column=column_name,
      values={"value": numeric_ks_statistic(source_series, synthetic_series)},
    ),
  ]


def _disclosure_risk_metrics(
  source_df: pl.DataFrame,
  synthetic_df: pl.DataFrame,
  schema: DatasetSchema,
  evaluated_columns: tuple[str, ...],
  *,
  nearest_neighbor_columns: Iterable[str] | None,
  max_pairwise_rows: int,
  random_seed: int,
) -> list[MetricResult]:
  metrics: list[MetricResult] = []
  source_risk_df = source_df.select(list(evaluated_columns))
  synthetic_risk_df = synthetic_df.select(list(evaluated_columns))

  metrics.extend(
    [
      MetricResult(
        category="disclosure_risk",
        name="exact_row_match_rate",
        values={"value": exact_row_match_rate(source_risk_df, synthetic_risk_df)},
        details={"columns": list(evaluated_columns)},
      ),
      MetricResult(
        category="disclosure_risk",
        name="source_duplicate_row_rate",
        values={"value": duplicate_row_rate(source_risk_df)},
        details={"columns": list(evaluated_columns)},
      ),
      MetricResult(
        category="disclosure_risk",
        name="synthetic_duplicate_row_rate",
        values={"value": duplicate_row_rate(synthetic_risk_df)},
        details={"columns": list(evaluated_columns)},
      ),
    ]
  )

  nearest_columns, skipped_nearest_columns = _nearest_neighbor_columns(
    schema,
    evaluated_columns,
    nearest_neighbor_columns,
  )
  nearest_values = nearest_neighbor_distance_summary(
    source_df,
    synthetic_df,
    nearest_columns,
    max_pairwise_rows=max_pairwise_rows,
    random_seed=random_seed,
  )
  metrics.append(
    MetricResult(
      category="disclosure_risk",
      name="nearest_neighbor_distance_summary",
      values=_metric_values(nearest_values),
      details={
        "columns": list(nearest_columns),
        "skipped_columns": list(skipped_nearest_columns),
        "distance": "scaled Euclidean distance from each synthetic row to nearest source row",
      },
    )
  )
  return metrics


def _nearest_neighbor_columns(
  schema: DatasetSchema,
  evaluated_columns: tuple[str, ...],
  requested_columns: Iterable[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
  schema_columns = schema.columns_by_name
  evaluated_set = set(evaluated_columns)
  if requested_columns is None:
    resolved = tuple(
      column_name
      for column_name in evaluated_columns
      if schema_columns[column_name].data_type in {"integer", "float"}
    )
    return resolved, ()

  resolved_list = []
  skipped_list = []
  for column_name in requested_columns:
    column_spec = schema_columns.get(column_name)
    if (
      column_name in evaluated_set
      and column_spec is not None
      and column_spec.data_type in {"integer", "float"}
    ):
      resolved_list.append(column_name)
    else:
      skipped_list.append(column_name)

  return tuple(resolved_list), tuple(skipped_list)


def _metric_values(
  values: dict[str, float | int | None],
) -> dict[str, MetricScalar]:
  return {key: value for key, value in values.items()}
