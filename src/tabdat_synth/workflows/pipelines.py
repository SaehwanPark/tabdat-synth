from __future__ import annotations

import numpy as np
import polars as pl

from tabdat_synth.core.probability import sigmoid
from tabdat_synth.core.sampling import (
  estimate_empirical_distribution,
  generate_responses,
  sample_empirical_distribution,
  sample_truncated_normal_from_summary,
  summarize_truncated_normal_source,
)
from tabdat_synth.core.statistics import calculate_weighted_sums
from tabdat_synth.core.validation import (
  resolve_incoming_ancestor_outputs,
  topologically_sort_steps,
  validate_dataframe_against_schema,
  validate_generation_spec,
  validate_required_columns,
)
from tabdat_synth.domain.contracts import (
  CategoricalModelStepArtifact,
  ConfiguredStepArtifact,
  EmpiricalStepArtifact,
  FittedSynthesizer,
  FittedStepArtifact,
  GenerationSpec,
  GenerationStep,
  TruncatedNormalStepArtifact,
)
from tabdat_synth.interfaces.io import (
  load_source_dataframe,
  prepare_source_dataframe,
)
from tabdat_synth.interfaces.models import (
  fit_categorical_model,
  sample_categorical_feature,
)
from tabdat_synth.version import __version__


def generate_from_spec(spec: GenerationSpec) -> pl.DataFrame:
  """Generate synthetic rows from a generation spec."""
  fitted = fit_synthesizer(spec)
  return generate_from_fitted(fitted)


def fit_synthesizer(spec: GenerationSpec) -> FittedSynthesizer:
  """Fit a reusable synthesizer bundle from a generation spec."""
  validate_generation_spec(spec)
  ordered_steps = topologically_sort_steps(spec)
  source_columns = tuple(
    dict.fromkeys(
      step.source_column for step in ordered_steps if step.source_column is not None
    )
  )

  source_df = load_source_dataframe(
    spec.source_path,
    spec.source_format,
    member=spec.source_member,
  )
  validate_required_columns(source_df, source_columns, "source dataframe")

  if source_columns:
    prepared_source_df = prepare_source_dataframe(
      source_df,
      spec.schema,
      columns=source_columns,
    )
    validate_dataframe_against_schema(
      prepared_source_df,
      spec.schema,
      source_columns,
      "prepared source dataframe",
    )
  else:
    prepared_source_df = pl.DataFrame()

  source_row_count = source_df.height
  source_generated_df = pl.DataFrame()
  step_artifacts: list[FittedStepArtifact] = []

  for step in ordered_steps:
    artifact, source_output = _fit_generation_step(
      step,
      prepared_source_df,
      source_generated_df,
      source_row_count,
      spec,
    )
    step_artifacts.append(artifact)
    if source_output is not None:
      source_generated_df = _append_output_series(source_generated_df, source_output)

  return FittedSynthesizer(
    spec=spec,
    step_artifacts=tuple(step_artifacts),
    package_version=__version__,
  )


def generate_from_fitted(
  fitted: FittedSynthesizer,
  *,
  n_samples: int | None = None,
  random_seed: int | None = None,
) -> pl.DataFrame:
  """Generate synthetic rows from a fitted synthesizer bundle."""
  validate_generation_spec(fitted.spec)
  _validate_fitted_synthesizer(fitted)

  resolved_n_samples = fitted.spec.n_samples if n_samples is None else n_samples
  resolved_seed = fitted.spec.random_seed if random_seed is None else random_seed
  if resolved_n_samples <= 0:
    raise ValueError("n_samples must be positive")

  synth_df = pl.DataFrame()

  for step_idx, artifact in enumerate(fitted.step_artifacts):
    output_series = _execute_fitted_step(
      artifact,
      synth_df,
      step_idx,
      fitted.spec,
      resolved_n_samples,
      resolved_seed,
    )
    synth_df = _append_output_series(synth_df, output_series)

  generated_columns = tuple(
    artifact.step.output_column for artifact in fitted.step_artifacts
  )
  validate_dataframe_against_schema(
    synth_df,
    fitted.spec.schema,
    generated_columns,
    "generated dataframe",
  )
  return synth_df


def _fit_generation_step(
  step: GenerationStep,
  prepared_source_df: pl.DataFrame,
  source_generated_df: pl.DataFrame,
  source_row_count: int,
  spec: GenerationSpec,
) -> tuple[FittedStepArtifact, pl.Series | None]:
  column_spec = spec.schema.columns_by_name[step.output_column]

  if step.kind == "empirical":
    assert step.source_column is not None
    source_series = prepared_source_df[step.source_column]
    values, probabilities = estimate_empirical_distribution(source_series)
    return (
      EmpiricalStepArtifact(
        step=step,
        values=tuple(values),
        probabilities=tuple(probabilities),
      ),
      _cast_output_series(step.output_column, source_series.to_list(), column_spec),
    )

  if step.kind == "truncated_normal":
    assert step.source_column is not None
    source_series = prepared_source_df[step.source_column]
    observed_mean, observed_std, minimum, maximum = summarize_truncated_normal_source(
      source_series,
      minimum=column_spec.minimum,
      maximum=column_spec.maximum,
    )
    return (
      TruncatedNormalStepArtifact(
        step=step,
        observed_mean=observed_mean,
        observed_std=observed_std,
        minimum=minimum,
        maximum=maximum,
        round_to_int=column_spec.data_type == "integer",
      ),
      _cast_output_series(step.output_column, source_series.to_list(), column_spec),
    )

  if step.kind == "categorical_model":
    assert step.source_column is not None
    source_series = prepared_source_df[step.source_column]
    input_columns = _resolve_categorical_input_columns(spec, step)
    X_source = source_generated_df.select(list(input_columns))
    model = fit_categorical_model(
      X_source,
      source_series,
      spec.model_backend,
      spec.random_seed,
    )
    feature_order = tuple(getattr(model, "_sdg_feature_order"))
    return (
      CategoricalModelStepArtifact(
        step=step,
        backend_kind=spec.model_backend.kind,
        input_columns=input_columns,
        feature_order=feature_order,
        learned_categories=tuple(str(value) for value in model.classes_),
        fitted_model=model,
      ),
      _cast_output_series(step.output_column, source_series.to_list(), column_spec),
    )

  if step.kind == "coefficient_sum":
    assert step.coefficient_sum is not None
    coefficient_values = _calculate_coefficient_sum_series(
      step,
      source_generated_df,
      source_row_count,
    )
    return ConfiguredStepArtifact(step=step), _cast_output_series(
      step.output_column,
      coefficient_values,
      column_spec,
    )

  if step.kind == "bin_map":
    binned_values = _calculate_bin_map_series(step, source_generated_df)
    return ConfiguredStepArtifact(step=step), _cast_output_series(
      step.output_column,
      binned_values,
      column_spec,
    )

  if step.kind == "sigmoid_probability":
    probability_values = _calculate_sigmoid_probability_series(
      step,
      source_generated_df,
    )
    return ConfiguredStepArtifact(step=step), _cast_output_series(
      step.output_column,
      probability_values,
      column_spec,
    )

  if step.kind == "bernoulli_response":
    assert step.response is not None
    return ConfiguredStepArtifact(step=step), None

  raise ValueError(f"unsupported step kind '{step.kind}'")


def _execute_fitted_step(
  artifact: FittedStepArtifact,
  synth_df: pl.DataFrame,
  step_idx: int,
  spec: GenerationSpec,
  n_samples: int,
  random_seed: int,
) -> pl.Series:
  step = artifact.step
  column_spec = spec.schema.columns_by_name[step.output_column]
  step_seed = random_seed + step_idx

  if isinstance(artifact, EmpiricalStepArtifact):
    sampled_values = sample_empirical_distribution(
      n_samples,
      artifact.values,
      artifact.probabilities,
      step_seed,
    )
    return _cast_output_series(step.output_column, sampled_values, column_spec)

  if isinstance(artifact, TruncatedNormalStepArtifact):
    sampled_values = sample_truncated_normal_from_summary(
      n_samples,
      artifact.observed_mean,
      artifact.observed_std,
      artifact.minimum,
      artifact.maximum,
      step_seed,
      mean_shift=step.mean_shift,
      round_to_int=artifact.round_to_int,
    )
    return _cast_output_series(step.output_column, sampled_values, column_spec)

  if isinstance(artifact, CategoricalModelStepArtifact):
    X_synth = synth_df.select(list(artifact.input_columns))
    sampled_categories = sample_categorical_feature(
      artifact.fitted_model,
      X_synth,
      list(artifact.learned_categories),
      step_idx,
      random_seed,
      feature_order=artifact.feature_order,
    )
    return _cast_output_series(step.output_column, sampled_categories, column_spec)

  if step.kind == "coefficient_sum":
    assert step.coefficient_sum is not None
    values = _calculate_coefficient_sum_series(step, synth_df, n_samples)
    return _cast_output_series(step.output_column, values, column_spec)

  if step.kind == "bin_map":
    values = _calculate_bin_map_series(step, synth_df)
    return _cast_output_series(step.output_column, values, column_spec)

  if step.kind == "sigmoid_probability":
    values = _calculate_sigmoid_probability_series(step, synth_df)
    return _cast_output_series(step.output_column, values, column_spec)

  if step.kind == "bernoulli_response":
    assert step.response is not None
    input_values = _input_values(step, synth_df)
    _, outcomes = generate_responses(
      input_values,
      step.response.intercept,
      step.response.slope,
      step_seed,
    )
    return _cast_output_series(step.output_column, outcomes, column_spec)

  raise ValueError(f"unsupported step kind '{step.kind}'")


def _calculate_coefficient_sum_series(
  step: GenerationStep,
  frame: pl.DataFrame,
  row_count: int,
) -> np.ndarray:
  if not step.parent_columns:
    assert step.coefficient_sum is not None
    return np.full(row_count, step.coefficient_sum.intercept, dtype=float)
  assert step.coefficient_sum is not None
  return calculate_weighted_sums(
    frame.select(list(step.parent_columns)),
    step.coefficient_sum,
  )


def _calculate_sigmoid_probability_series(
  step: GenerationStep,
  frame: pl.DataFrame,
) -> np.ndarray:
  return sigmoid(_input_values(step, frame))


def _calculate_bin_map_series(
  step: GenerationStep,
  frame: pl.DataFrame,
) -> np.ndarray:
  assert step.bin_map is not None
  assert step.input_column is not None
  breaks = list(step.bin_map.breaks)
  labels = list(step.bin_map.labels)
  binned = (
    frame.select(
      pl.col(step.input_column)
      .cut(breaks=breaks, labels=labels)
      .cast(pl.String)
      .alias(step.output_column)
    )
    .to_series()
    .to_numpy()
  )
  return binned


def _resolve_categorical_input_columns(
  spec: GenerationSpec,
  step: GenerationStep,
) -> tuple[str, ...]:
  if step.feature_columns:
    return step.feature_columns
  return resolve_incoming_ancestor_outputs(spec, step)


def _input_values(step: GenerationStep, frame: pl.DataFrame) -> np.ndarray:
  assert step.input_column is not None
  return frame[step.input_column].cast(pl.Float64).to_numpy()


def _append_output_series(df: pl.DataFrame, series: pl.Series) -> pl.DataFrame:
  if df.width == 0:
    return pl.DataFrame({series.name: series})
  return df.with_columns(series)


def _validate_fitted_synthesizer(fitted: FittedSynthesizer) -> None:
  ordered_steps = topologically_sort_steps(fitted.spec)
  if len(ordered_steps) != len(fitted.step_artifacts):
    raise ValueError(
      "fitted synthesizer step count does not match the validated generation graph"
    )

  for expected_step, artifact in zip(ordered_steps, fitted.step_artifacts, strict=True):
    if artifact.step != expected_step:
      raise ValueError(
        "fitted synthesizer step order does not match the validated generation graph"
      )


def _cast_output_series(
  output_name: str,
  values: np.ndarray | list[str] | list[int] | list[float],
  column_spec,
) -> pl.Series:
  output_series = pl.Series(name=output_name, values=values)
  if column_spec.data_type == "categorical":
    return output_series.cast(pl.String).cast(pl.Categorical)
  if column_spec.data_type == "integer":
    return output_series.cast(pl.Int64)
  return output_series.cast(pl.Float64)
