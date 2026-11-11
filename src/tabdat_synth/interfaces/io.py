from __future__ import annotations

import io
import os
from pathlib import Path
import zipfile
from typing import Any, cast

import polars as pl
import yaml  # type: ignore[import-untyped]

from tabdat_synth.core.sampling import MISSING_CATEGORY_LABEL
from tabdat_synth.domain.contracts import (
  BernoulliResponseSpec,
  BinMapSpec,
  CoefficientSumSpec,
  CoefficientTerm,
  ColumnSpec,
  DatasetSchema,
  GenerationSpec,
  GenerationStep,
  ModelBackendSpec,
)

_LEGACY_RESPONSE_COLUMNS = ("Hosp_1y", "Mort_1y")
_LEGACY_RISK_COLUMNS = ("pHosp_1y", "pMort_1y")
_LEGACY_AGE_COLUMNS = ("age_cont", "agegrp")
_LEGACY_VA_DROP_COLUMNS = ("patienticn", "yr", "Death1y", "intercept")
_SOURCE_REQUIRED_STEP_KINDS = {"empirical", "truncated_normal", "categorical_model"}


def resolve_va_data_dir(env_key: str = "VA_DIR", load_env: bool = True) -> Path:
  """Resolve the configured VA data directory at the IO boundary."""
  if load_env:
    _load_dotenv_if_available()

  value = os.getenv(env_key)
  if not value:
    raise ValueError(f"{env_key} is not configured")

  return Path(value)


def _load_dotenv_if_available() -> None:
  try:
    from dotenv import load_dotenv
  except ModuleNotFoundError as exc:
    if exc.name == "dotenv":
      return
    raise

  load_dotenv()


def load_and_prepare_data(
  data_path: Path, frac: float, seed: int
) -> tuple[pl.DataFrame, list[str]]:
  """Load and prepare the legacy VA parquet dataset for generation."""
  df = (
    pl.read_parquet(data_path)
    .drop(*_LEGACY_VA_DROP_COLUMNS)
    .sample(fraction=frac, seed=seed)
  )

  target_cols = [
    col
    for col in df.schema.names()
    if col
    not in (
      list(_LEGACY_RESPONSE_COLUMNS)
      + list(_LEGACY_RISK_COLUMNS)
      + list(_LEGACY_AGE_COLUMNS)
    )
  ]

  expr_list = [
    pl.col(col).cast(pl.String).cast(pl.Categorical)
    for col in target_cols + ["agegrp"]
  ]
  df_prepared = df.with_columns(expr_list)

  return df_prepared, target_cols


def load_generation_spec(path: Path | str) -> GenerationSpec:
  """Load a YAML generation spec into typed generation contracts."""
  config_path = Path(path).resolve()
  raw = yaml.safe_load(config_path.read_text())
  return load_generation_spec_mapping(raw, base_path=config_path.parent)


def load_generation_spec_mapping(
  raw: object,
  *,
  base_path: Path | None = None,
) -> GenerationSpec:
  """Load a raw generation-spec mapping into typed contracts."""
  if not isinstance(raw, dict):
    raise ValueError("generation spec must be a mapping")
  if "va_outcomes" in raw:
    raise ValueError(
      "generation spec 'va_outcomes' is no longer supported; "
      "use coefficient_sum, sigmoid_probability, and bernoulli_response steps"
    )

  source = raw.get("source", {})
  if not isinstance(source, dict):
    raise ValueError("generation spec 'source' must be a mapping")

  raw_source_path = source.get("path")
  if not raw_source_path:
    raise ValueError("generation spec source.path is required")

  schema = _load_dataset_schema(raw.get("schema"))
  steps = _load_generation_steps(raw.get("steps", []))
  model_backend = _load_model_backend(raw.get("model_backend"), raw)
  resolved_source_path = _resolve_source_path(raw_source_path, base_path)

  return GenerationSpec(
    name=raw["name"],
    source_path=resolved_source_path,
    source_format=source["format"],
    source_member=source.get("member"),
    schema=schema,
    steps=steps,
    model_backend=model_backend,
    n_samples=int(raw.get("n_samples", 100_000)),
    random_seed=int(raw.get("random_seed", 343)),
  )


def generation_spec_to_dict(spec: GenerationSpec) -> dict[str, object]:
  """Convert a typed generation spec into a normalized mapping."""
  source: dict[str, object] = {
    "path": str(spec.source_path),
    "format": spec.source_format,
  }
  if spec.source_member is not None:
    source["member"] = spec.source_member

  return {
    "name": spec.name,
    "source": source,
    "n_samples": spec.n_samples,
    "random_seed": spec.random_seed,
    "model_backend": {
      "kind": spec.model_backend.kind,
      "params": dict(spec.model_backend.params),
    },
    "schema": {
      "columns": [_column_spec_to_dict(column) for column in spec.schema.columns]
    },
    "steps": [_generation_step_to_dict(step) for step in spec.steps],
  }


def load_source_dataframe(
  path: Path, source_format: str, member: str | None = None
) -> pl.DataFrame:
  """Load a generic source dataframe from parquet, csv, or zip-wrapped csv."""
  if source_format == "parquet":
    return pl.read_parquet(path)
  if source_format == "csv":
    return pl.read_csv(path, null_values="")
  if source_format == "zip_csv":
    with zipfile.ZipFile(path) as archive:
      member_name = member or _infer_zip_member(archive, path)
      with archive.open(member_name) as zipped_file:
        return pl.read_csv(io.BytesIO(zipped_file.read()), null_values="")

  raise ValueError(f"unsupported source_format '{source_format}'")


def prepare_source_dataframe(
  df: pl.DataFrame,
  schema: DatasetSchema,
  columns: tuple[str, ...] | None = None,
) -> pl.DataFrame:
  """Apply schema-driven casts and missing-value policies to source data."""
  required_columns = columns or tuple(schema.columns_by_name)
  prepared = df.select(list(required_columns))

  for column_name in required_columns:
    column_spec = schema.columns_by_name[column_name]
    prepared = prepared.with_columns(_prepare_column(prepared, column_spec))

  return prepared


def _prepare_column(df: pl.DataFrame, column_spec: ColumnSpec) -> pl.Expr:
  expr = pl.col(column_spec.name)

  if column_spec.data_type == "categorical":
    expr = expr.cast(pl.String, strict=False)
    if column_spec.missing_value_policy == "as_category":
      expr = expr.fill_null(MISSING_CATEGORY_LABEL)
    return expr.cast(pl.Categorical)

  if column_spec.data_type == "integer":
    expr = expr.cast(pl.Int64, strict=False)
    if column_spec.missing_value_policy == "mean_impute":
      integer_fill_value = _integer_mean_fill(df[column_spec.name])
      expr = expr.fill_null(integer_fill_value)
    return expr.cast(pl.Int64)

  expr = expr.cast(pl.Float64, strict=False)
  if column_spec.missing_value_policy == "mean_impute":
    float_fill_value: float = _float_mean_fill(df[column_spec.name])
    expr = expr.fill_null(float_fill_value)
  return expr.cast(pl.Float64)


def _integer_mean_fill(series: pl.Series) -> int:
  mean_value: object = series.cast(pl.Float64, strict=False).mean()
  if mean_value is None:
    return 0
  return int(round(float(cast(Any, mean_value))))


def _float_mean_fill(series: pl.Series) -> float:
  mean_value: object = series.cast(pl.Float64, strict=False).mean()
  if mean_value is None:
    return 0.0
  return float(cast(Any, mean_value))


def _infer_zip_member(archive: zipfile.ZipFile, path: Path) -> str:
  members = [name for name in archive.namelist() if not name.endswith("/")]
  if len(members) != 1:
    raise ValueError(f"zip_csv source '{path}' requires an explicit member")
  return members[0]


def _resolve_source_path(raw_source_path: str, base_path: Path | None) -> Path:
  source_path = Path(raw_source_path)
  if source_path.is_absolute():
    return source_path
  if base_path is not None:
    return (base_path / source_path).resolve()
  return source_path.resolve()


def _load_dataset_schema(raw_schema: object) -> DatasetSchema:
  if not isinstance(raw_schema, dict):
    raise ValueError("generation spec schema must be a mapping")

  raw_columns = raw_schema.get("columns", [])
  if not isinstance(raw_columns, list):
    raise ValueError("generation spec schema.columns must be a list")

  columns = []
  for raw_column in raw_columns:
    if not isinstance(raw_column, dict):
      raise ValueError("schema column definitions must be mappings")
    columns.append(
      ColumnSpec(
        name=raw_column["name"],
        data_type=raw_column["data_type"],
        role=raw_column.get("role", "feature"),
        nullable=bool(raw_column.get("nullable", False)),
        missing_value_policy=raw_column.get("missing_value_policy", "forbid"),
        allowed_values=tuple(
          str(value) for value in raw_column.get("allowed_values", [])
        ),
        minimum=raw_column.get("minimum"),
        maximum=raw_column.get("maximum"),
      )
    )

  return DatasetSchema(columns=tuple(columns))


def _load_generation_steps(raw_steps: object) -> tuple[GenerationStep, ...]:
  if not isinstance(raw_steps, list):
    raise ValueError("generation spec steps must be a list")

  steps = []
  for raw_step in raw_steps:
    if not isinstance(raw_step, dict):
      raise ValueError("generation steps must be mappings")

    kind = raw_step["kind"]
    source_column = raw_step.get("source_column")
    if source_column is None and kind in _SOURCE_REQUIRED_STEP_KINDS:
      source_column = raw_step["output_column"]

    steps.append(
      GenerationStep(
        name=raw_step["name"],
        kind=kind,
        output_column=raw_step["output_column"],
        source_column=source_column,
        parent_columns=tuple(raw_step.get("parent_columns", [])),
        input_column=raw_step.get("input_column"),
        mean_shift=float(cast(Any, raw_step.get("mean_shift", 0.0))),
        coefficient_sum=_load_coefficient_sum(raw_step.get("coefficient_sum")),
        response=_load_response(raw_step.get("response")),
        feature_columns=tuple(
          str(column) for column in raw_step.get("feature_columns", [])
        ),
        bin_map=_load_bin_map(raw_step.get("bin_map")),
      )
    )

  return tuple(steps)


def _load_model_backend(
  raw_model_backend: object,
  raw_spec: dict[str, object],
) -> ModelBackendSpec:
  if raw_model_backend is None:
    n_estimators_raw = raw_spec.get("lgb_n_estimators", 100)
    verbosity_raw = raw_spec.get("lgb_verbosity", -1)
    return ModelBackendSpec(
      kind="lightgbm",
      params={
        "n_estimators": int(cast(Any, n_estimators_raw)),
        "verbosity": int(cast(Any, verbosity_raw)),
      },
    )

  if not isinstance(raw_model_backend, dict):
    raise ValueError("model_backend must be a mapping when provided")

  raw_params = raw_model_backend.get("params", {})
  if not isinstance(raw_params, dict):
    raise ValueError("model_backend.params must be a mapping when provided")

  return ModelBackendSpec(
    kind=str(raw_model_backend.get("kind", "")).strip(),
    params={str(key): value for key, value in raw_params.items()},
  )


def _load_coefficient_sum(raw_coefficient_sum: object) -> CoefficientSumSpec | None:
  if raw_coefficient_sum is None:
    return None
  if not isinstance(raw_coefficient_sum, dict):
    raise ValueError("coefficient_sum must be a mapping when provided")

  raw_terms = raw_coefficient_sum.get("terms", [])
  if not isinstance(raw_terms, list):
    raise ValueError("coefficient_sum.terms must be a list when provided")

  terms = []
  for raw_term in raw_terms:
    if not isinstance(raw_term, dict):
      raise ValueError("coefficient_sum terms must be mappings")
    terms.append(
      CoefficientTerm(
        feature_name=raw_term["feature_name"],
        coefficient=float(cast(Any, raw_term["coefficient"])),
        value=str(raw_term["value"]) if "value" in raw_term else None,
      )
    )

  return CoefficientSumSpec(
    intercept=float(cast(Any, raw_coefficient_sum.get("intercept", 0.0))),
    reference_value=str(raw_coefficient_sum.get("reference_value", "99")),
    terms=tuple(terms),
  )


def _load_response(raw_response: object) -> BernoulliResponseSpec | None:
  if raw_response is None:
    return None
  if not isinstance(raw_response, dict):
    raise ValueError("response must be a mapping when provided")

  return BernoulliResponseSpec(
    intercept=float(cast(Any, raw_response.get("intercept", 0.0))),
    slope=float(cast(Any, raw_response.get("slope", 1.0))),
  )


def _load_bin_map(raw_bin_map: object) -> BinMapSpec | None:
  if raw_bin_map is None:
    return None
  if not isinstance(raw_bin_map, dict):
    raise ValueError("bin_map must be a mapping when provided")

  raw_breaks = raw_bin_map.get("breaks", [])
  raw_labels = raw_bin_map.get("labels", [])
  if not isinstance(raw_breaks, list):
    raise ValueError("bin_map.breaks must be a list when provided")
  if not isinstance(raw_labels, list):
    raise ValueError("bin_map.labels must be a list when provided")

  return BinMapSpec(
    breaks=tuple(float(cast(Any, value)) for value in raw_breaks),
    labels=tuple(str(value) for value in raw_labels),
  )


def _column_spec_to_dict(column: ColumnSpec) -> dict[str, object]:
  payload: dict[str, object] = {
    "name": column.name,
    "data_type": column.data_type,
    "role": column.role,
    "nullable": column.nullable,
    "missing_value_policy": column.missing_value_policy,
  }
  if column.allowed_values:
    payload["allowed_values"] = list(column.allowed_values)
  if column.minimum is not None:
    payload["minimum"] = column.minimum
  if column.maximum is not None:
    payload["maximum"] = column.maximum
  return payload


def _generation_step_to_dict(step: GenerationStep) -> dict[str, object]:
  payload: dict[str, object] = {
    "name": step.name,
    "kind": step.kind,
    "output_column": step.output_column,
  }
  if step.source_column is not None and step.source_column != step.output_column:
    payload["source_column"] = step.source_column
  if step.source_column is not None and step.kind in _SOURCE_REQUIRED_STEP_KINDS and step.source_column == step.output_column:
    pass
  if step.parent_columns:
    payload["parent_columns"] = list(step.parent_columns)
  if step.input_column is not None:
    payload["input_column"] = step.input_column
  if step.mean_shift != 0.0:
    payload["mean_shift"] = step.mean_shift
  if step.coefficient_sum is not None:
    payload["coefficient_sum"] = _coefficient_sum_to_dict(step.coefficient_sum)
  if step.response is not None:
    payload["response"] = _response_to_dict(step.response)
  if step.feature_columns:
    payload["feature_columns"] = list(step.feature_columns)
  if step.bin_map is not None:
    payload["bin_map"] = _bin_map_to_dict(step.bin_map)
  return payload


def _coefficient_sum_to_dict(coefficient_sum: CoefficientSumSpec) -> dict[str, object]:
  payload: dict[str, object] = {
    "intercept": coefficient_sum.intercept,
    "reference_value": coefficient_sum.reference_value,
    "terms": [],
  }
  payload["terms"] = [
    _coefficient_term_to_dict(term) for term in coefficient_sum.terms
  ]
  return payload


def _coefficient_term_to_dict(term: CoefficientTerm) -> dict[str, object]:
  payload: dict[str, object] = {
    "feature_name": term.feature_name,
    "coefficient": term.coefficient,
  }
  if term.value is not None:
    payload["value"] = term.value
  return payload


def _response_to_dict(response: BernoulliResponseSpec) -> dict[str, object]:
  return {
    "intercept": response.intercept,
    "slope": response.slope,
  }


def _bin_map_to_dict(bin_map: BinMapSpec) -> dict[str, object]:
  return {
    "breaks": list(bin_map.breaks),
    "labels": list(bin_map.labels),
  }
