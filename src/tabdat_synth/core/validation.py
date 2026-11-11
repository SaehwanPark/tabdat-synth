from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any, cast

import polars as pl

from tabdat_synth.domain.contracts import (
  BernoulliResponseSpec,
  BinMapSpec,
  CoefficientSumSpec,
  ColumnSpec,
  DatasetSchema,
  GenerationSpec,
  GenerationStep,
  ModelBackendSpec,
)

ALLOWED_MISSING_VALUE_POLICIES = {
  "categorical": {"forbid", "as_category"},
  "integer": {"forbid", "mean_impute"},
  "float": {"forbid", "mean_impute"},
}
_SOURCE_STEP_KINDS = {"empirical", "truncated_normal", "categorical_model"}
_FEATURE_STEP_KINDS = {"categorical_model", "coefficient_sum"}
_INPUT_STEP_KINDS = {"sigmoid_probability", "bernoulli_response"}
_BIN_MAP_STEP_KINDS = {"bin_map"}


def validate_generation_spec(spec: GenerationSpec) -> None:
  """Validate the declarative generation spec."""
  if spec.n_samples <= 0:
    raise ValueError("n_samples must be positive")

  validate_dataset_schema(spec.schema)
  validate_model_backend_spec(spec.model_backend)

  schema_columns = spec.schema.columns_by_name
  step_outputs: dict[str, GenerationStep] = {}

  for step in spec.steps:
    if step.output_column in step_outputs:
      raise ValueError(f"duplicate output column '{step.output_column}' in steps")
    if step.output_column not in schema_columns:
      raise ValueError(f"unknown output column '{step.output_column}'")

    output_spec = schema_columns[step.output_column]
    if output_spec.role == "identifier":
      raise ValueError(
        f"output column '{step.output_column}' cannot use identifier role"
      )

    _validate_step_shape(step, schema_columns)
    step_outputs[step.output_column] = step

  for step in spec.steps:
    for dependency in step_dependencies(step):
      if dependency not in step_outputs:
        raise ValueError(
          f"unknown dependency column '{dependency}' in step '{step.name}'"
        )
      if dependency == step.output_column:
        raise ValueError(f"step '{step.name}' cannot depend on itself")

  _validate_feature_columns(spec)
  _validate_terminal_bernoulli_steps(spec)
  topologically_sort_steps(spec)


def validate_model_backend_spec(model_backend: ModelBackendSpec) -> None:
  """Validate built-in model-backend configuration."""
  if not model_backend.kind:
    raise ValueError("model_backend.kind is required")

  params = model_backend.params
  if model_backend.kind == "lightgbm":
    _validate_allowed_param_keys(
      params,
      allowed={"n_estimators", "verbosity"},
      context="lightgbm",
    )
    n_estimators = params.get("n_estimators", 100)
    if int(n_estimators) <= 0:
      raise ValueError("lightgbm model_backend.params.n_estimators must be positive")
    return

  if model_backend.kind == "logistic_regression":
    _validate_allowed_param_keys(
      params,
      allowed={"C", "max_iter", "solver"},
      context="logistic_regression",
    )
    max_iter = params.get("max_iter", 500)
    if int(max_iter) <= 0:
      raise ValueError(
        "logistic_regression model_backend.params.max_iter must be positive"
      )
    c_value = params.get("C", 1.0)
    if float(c_value) <= 0:
      raise ValueError("logistic_regression model_backend.params.C must be positive")


def _validate_allowed_param_keys(
  params: Mapping[str, object],
  *,
  allowed: set[str],
  context: str,
) -> None:
  unexpected = sorted(set(params) - allowed)
  if unexpected:
    raise ValueError(
      f"{context} model_backend.params contains unsupported keys: {unexpected}"
    )


def validate_dataset_schema(schema: DatasetSchema) -> None:
  """Validate static dataset schema declarations."""
  seen: set[str] = set()
  for column in schema.columns:
    if column.name in seen:
      raise ValueError(f"duplicate schema column '{column.name}'")
    seen.add(column.name)
    _validate_column_spec(column)


def _validate_column_spec(column: ColumnSpec) -> None:
  allowed_policies = ALLOWED_MISSING_VALUE_POLICIES[column.data_type]
  if column.missing_value_policy not in allowed_policies:
    raise ValueError(
      f"column '{column.name}' uses invalid missing_value_policy "
      f"'{column.missing_value_policy}' for {column.data_type}"
    )

  if column.data_type == "categorical":
    if column.minimum is not None or column.maximum is not None:
      raise ValueError(
        f"categorical column '{column.name}' cannot declare numeric bounds"
      )
    if len(set(column.allowed_values)) != len(column.allowed_values):
      raise ValueError(f"column '{column.name}' has duplicate allowed_values")
  else:
    if column.allowed_values:
      raise ValueError(
        f"numeric column '{column.name}' cannot declare allowed_values"
      )
    if (
      column.minimum is not None
      and column.maximum is not None
      and column.minimum > column.maximum
    ):
      raise ValueError(f"column '{column.name}' has minimum greater than maximum")


def _validate_step_shape(
  step: GenerationStep,
  schema_columns: dict[str, ColumnSpec],
) -> None:
  output_spec = schema_columns[step.output_column]

  if step.kind in _SOURCE_STEP_KINDS:
    if step.source_column is None:
      raise ValueError(f"{step.kind} step '{step.name}' requires source_column")
    if step.source_column not in schema_columns:
      raise ValueError(f"unknown source column '{step.source_column}'")
  elif step.source_column is not None:
    raise ValueError(f"{step.kind} step '{step.name}' cannot declare source_column")

  if step.kind in _INPUT_STEP_KINDS or step.kind in _BIN_MAP_STEP_KINDS:
    if step.input_column is None:
      raise ValueError(f"{step.kind} step '{step.name}' requires input_column")
  elif step.input_column is not None:
    raise ValueError(f"{step.kind} step '{step.name}' cannot declare input_column")

  if step.kind == "bin_map":
    _validate_bin_map_step(step, output_spec)
  elif step.bin_map is not None:
    raise ValueError(f"{step.kind} step '{step.name}' cannot declare bin_map")

  if step.kind == "categorical_model":
    if not step.parent_columns:
      raise ValueError(
        f"categorical_model step '{step.name}' requires parent_columns"
      )
    if output_spec.data_type != "categorical":
      raise ValueError(
        f"categorical_model step '{step.name}' requires categorical output"
      )
  elif step.kind == "coefficient_sum":
    _validate_coefficient_sum_step(step, output_spec)
  elif step.parent_columns:
    raise ValueError(f"{step.kind} step '{step.name}' cannot declare parent_columns")

  if step.feature_columns and step.kind != "categorical_model":
    raise ValueError(
      f"{step.kind} step '{step.name}' cannot declare feature_columns"
    )

  if step.kind == "truncated_normal" and output_spec.data_type == "categorical":
    raise ValueError(
      f"truncated_normal step '{step.name}' cannot target a categorical column"
    )

  if step.kind == "sigmoid_probability":
    if output_spec.data_type != "float":
      raise ValueError(
        f"sigmoid_probability step '{step.name}' requires float output"
      )
    _validate_forbidden_specs(step, response=False, coefficient_sum=False)

  if step.kind == "bernoulli_response":
    if output_spec.data_type != "integer":
      raise ValueError(
        f"bernoulli_response step '{step.name}' requires integer output"
      )
    if step.response is None:
      raise ValueError(f"bernoulli_response step '{step.name}' requires response")
    _validate_response_spec(step.name, step.response)
    if step.coefficient_sum is not None:
      raise ValueError(
        f"bernoulli_response step '{step.name}' cannot declare coefficient_sum"
      )

  if step.kind in {"empirical", "truncated_normal", "categorical_model"}:
    _validate_forbidden_specs(step, response=True, coefficient_sum=True)


def _validate_coefficient_sum_step(
  step: GenerationStep,
  output_spec: ColumnSpec,
) -> None:
  if output_spec.data_type != "float":
    raise ValueError(f"coefficient_sum step '{step.name}' requires float output")
  if step.coefficient_sum is None:
    raise ValueError(f"coefficient_sum step '{step.name}' requires coefficient_sum")
  if step.input_column is not None:
    raise ValueError(f"coefficient_sum step '{step.name}' cannot declare input_column")
  if step.response is not None:
    raise ValueError(f"coefficient_sum step '{step.name}' cannot declare response")

  _validate_coefficient_sum_spec(step.name, step.coefficient_sum, step.parent_columns)


def _validate_coefficient_sum_spec(
  step_name: str,
  coefficient_sum: CoefficientSumSpec,
  parent_columns: tuple[str, ...],
) -> None:
  for term in coefficient_sum.terms:
    if term.feature_name not in parent_columns:
      raise ValueError(
        f"coefficient_sum step '{step_name}' term references "
        f"unknown parent feature '{term.feature_name}'"
      )
    if term.value == coefficient_sum.reference_value:
      raise ValueError(
        f"coefficient_sum step '{step_name}' cannot assign a coefficient to "
        f"the reference value '{coefficient_sum.reference_value}'"
      )


def _validate_response_spec(step_name: str, response: BernoulliResponseSpec) -> None:
  if response.slope <= 0:
    raise ValueError(
      f"bernoulli_response step '{step_name}' requires positive response.slope"
    )


def _validate_feature_columns(spec: GenerationSpec) -> None:
  for step in spec.steps:
    if not step.feature_columns:
      continue
    if len(set(step.feature_columns)) != len(step.feature_columns):
      raise ValueError(f"step '{step.name}' has duplicate feature_columns")
    ancestors = resolve_incoming_ancestor_outputs(spec, step)
    unknown = [column for column in step.feature_columns if column not in ancestors]
    if unknown:
      raise ValueError(
        f"step '{step.name}' feature_columns must already be ancestor outputs; "
        f"unknown feature columns: {unknown}"
      )


def _validate_bin_map_step(step: GenerationStep, output_spec: ColumnSpec) -> None:
  if step.source_column is not None:
    raise ValueError(f"bin_map step '{step.name}' cannot declare source_column")
  if step.parent_columns:
    raise ValueError(f"bin_map step '{step.name}' cannot declare parent_columns")
  if step.response is not None:
    raise ValueError(f"bin_map step '{step.name}' cannot declare response")
  if step.coefficient_sum is not None:
    raise ValueError(f"bin_map step '{step.name}' cannot declare coefficient_sum")
  if step.feature_columns:
    raise ValueError(f"bin_map step '{step.name}' cannot declare feature_columns")
  if output_spec.data_type != "categorical":
    raise ValueError(f"bin_map step '{step.name}' requires categorical output")
  if step.bin_map is None:
    raise ValueError(f"bin_map step '{step.name}' requires bin_map")
  _validate_bin_map_spec(step.name, step.bin_map)


def _validate_bin_map_spec(step_name: str, bin_map: BinMapSpec) -> None:
  breaks = bin_map.breaks
  labels = bin_map.labels

  if not breaks:
    raise ValueError(f"bin_map step '{step_name}' requires at least one break")
  if len(labels) != len(breaks) + 1:
    raise ValueError(
      f"bin_map step '{step_name}' requires len(labels) == len(breaks) + 1"
    )
  if any(right <= left for left, right in zip(breaks, breaks[1:])):
    raise ValueError(f"bin_map step '{step_name}' requires strictly increasing breaks")


def _validate_forbidden_specs(
  step: GenerationStep,
  *,
  response: bool,
  coefficient_sum: bool,
) -> None:
  if response and step.response is not None:
    raise ValueError(f"{step.kind} step '{step.name}' cannot declare response")
  if coefficient_sum and step.coefficient_sum is not None:
    raise ValueError(
      f"{step.kind} step '{step.name}' cannot declare coefficient_sum"
    )


def _validate_terminal_bernoulli_steps(spec: GenerationSpec) -> None:
  bernoulli_outputs = {
    step.output_column
    for step in spec.steps
    if step.kind == "bernoulli_response"
  }
  if not bernoulli_outputs:
    return

  for step in spec.steps:
    for dependency in step_dependencies(step):
      if dependency in bernoulli_outputs:
        raise ValueError(
          "bernoulli_response outputs must be terminal and cannot be used as "
          f"dependencies; step '{step.name}' depends on '{dependency}'"
        )


def step_dependencies(step: GenerationStep) -> tuple[str, ...]:
  """Return the declared DAG dependencies for one step."""
  dependencies = list(step.parent_columns)
  if step.input_column is not None:
    dependencies.append(step.input_column)
  return tuple(dependencies)


def topologically_sort_steps(spec: GenerationSpec) -> list[GenerationStep]:
  """Return generation steps in dependency order."""
  step_by_output = {step.output_column: step for step in spec.steps}
  indegree = {step.output_column: len(step_dependencies(step)) for step in spec.steps}
  children: dict[str, list[str]] = {step.output_column: [] for step in spec.steps}

  for step in spec.steps:
    for dependency in step_dependencies(step):
      children[dependency].append(step.output_column)

  ready = deque(
    step.output_column
    for step in spec.steps
    if indegree[step.output_column] == 0
  )
  ordered: list[GenerationStep] = []

  while ready:
    output_name = ready.popleft()
    ordered.append(step_by_output[output_name])
    for child in children[output_name]:
      indegree[child] -= 1
      if indegree[child] == 0:
        ready.append(child)

  if len(ordered) != len(spec.steps):
    raise ValueError("generation step graph contains a cycle")

  return ordered


def resolve_incoming_ancestor_outputs(
  spec: GenerationSpec,
  step: GenerationStep,
) -> tuple[str, ...]:
  """Return all ancestor output columns in stable topological order."""
  ordered_steps = topologically_sort_steps(spec)
  step_by_output = {candidate.output_column: candidate for candidate in spec.steps}

  pending = list(step.parent_columns)
  ancestor_set: set[str] = set()

  while pending:
    output_name = pending.pop()
    if output_name in ancestor_set:
      continue
    ancestor_set.add(output_name)
    pending.extend(step_dependencies(step_by_output[output_name]))

  return tuple(
    candidate.output_column
    for candidate in ordered_steps
    if candidate.output_column in ancestor_set
  )


def validate_required_columns(
  df: pl.DataFrame, required_columns: Iterable[str], context: str
) -> None:
  """Ensure required columns exist in a dataframe before use."""
  missing = [column for column in required_columns if column not in df.columns]
  if missing:
    raise ValueError(f"{context} is missing required columns: {missing}")


def validate_dataframe_against_schema(
  df: pl.DataFrame,
  schema: DatasetSchema,
  columns: Iterable[str],
  context: str,
) -> None:
  """Validate runtime dataframe values against schema constraints."""
  validate_required_columns(df, columns, context)

  for column_name in columns:
    column_spec = schema.columns_by_name[column_name]
    series = df[column_name]

    if not column_spec.nullable and series.null_count() > 0:
      raise ValueError(f"{context} column '{column_name}' contains null values")

    non_null = series.drop_nulls()
    if non_null.is_empty():
      continue

    if column_spec.data_type == "categorical" and column_spec.allowed_values:
      observed = set(non_null.cast(pl.String).to_list())
      allowed = set(column_spec.allowed_values)
      unexpected = sorted(observed - allowed)
      if unexpected:
        raise ValueError(
          f"{context} column '{column_name}' has values outside allowed_values: "
          f"{unexpected}"
        )

    if column_spec.data_type in {"integer", "float"}:
      numeric = non_null.cast(pl.Float64)
      observed_min_raw = numeric.min()
      observed_max_raw = numeric.max()
      observed_min = (
        None if observed_min_raw is None else float(cast(Any, observed_min_raw))
      )
      observed_max = (
        None if observed_max_raw is None else float(cast(Any, observed_max_raw))
      )
      if (
        column_spec.minimum is not None
        and observed_min is not None
        and observed_min < column_spec.minimum
      ):
        raise ValueError(
          f"{context} column '{column_name}' falls below minimum "
          f"{column_spec.minimum}"
        )
      if (
        column_spec.maximum is not None
        and observed_max is not None
        and observed_max > column_spec.maximum
      ):
        raise ValueError(
          f"{context} column '{column_name}' exceeds maximum "
          f"{column_spec.maximum}"
        )
