from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias


ColumnDataType = Literal["categorical", "integer", "float"]
ColumnRole = Literal["feature", "outcome", "identifier"]
MissingValuePolicy = Literal["forbid", "as_category", "mean_impute"]
SourceFormat = Literal["parquet", "csv", "zip_csv"]
GenerationStepKind = Literal[
  "empirical",
  "truncated_normal",
  "categorical_model",
  "coefficient_sum",
  "sigmoid_probability",
  "bernoulli_response",
  "bin_map",
]
BackendParamValue: TypeAlias = str | int | float | bool
BackendScalarValue: TypeAlias = str | int | float
MetricScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class FreqShift:
  """Categorical frequency shift in percentage points."""

  source: str
  dest: str
  shift_pct: float


@dataclass(frozen=True)
class ModelBackendSpec:
  """Configuration for the categorical-model backend used by a spec."""

  kind: str = "lightgbm"
  params: dict[str, BackendParamValue] = field(default_factory=dict)


@dataclass(frozen=True)
class CoefficientTerm:
  """One term in a coefficient-based weighted sum."""

  feature_name: str
  coefficient: float
  value: str | None = None


@dataclass(frozen=True)
class CoefficientSumSpec:
  """Typed configuration for coefficient-based weighted sums."""

  intercept: float = 0.0
  reference_value: str = "99"
  terms: tuple[CoefficientTerm, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BernoulliResponseSpec:
  """Typed response-adjustment configuration for Bernoulli sampling."""

  intercept: float = 0.0
  slope: float = 1.0


@dataclass(frozen=True)
class BinMapSpec:
  """Typed deterministic binning configuration for one numeric input column."""

  breaks: tuple[float, ...] = field(default_factory=tuple)
  labels: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ColumnSpec:
  """Declarative schema metadata for one source or generated column."""

  name: str
  data_type: ColumnDataType
  role: ColumnRole = "feature"
  nullable: bool = False
  missing_value_policy: MissingValuePolicy = "forbid"
  allowed_values: tuple[str, ...] = field(default_factory=tuple)
  minimum: float | None = None
  maximum: float | None = None


@dataclass(frozen=True)
class DatasetSchema:
  """Schema contract for a configurable source dataset."""

  columns: tuple[ColumnSpec, ...]

  @property
  def columns_by_name(self) -> dict[str, ColumnSpec]:
    return {column.name: column for column in self.columns}


@dataclass(frozen=True)
class GenerationStep:
  """Declarative description of one synthetic feature-generation step."""

  name: str
  kind: GenerationStepKind
  output_column: str
  source_column: str | None = None
  parent_columns: tuple[str, ...] = field(default_factory=tuple)
  input_column: str | None = None
  mean_shift: float = 0.0
  coefficient_sum: CoefficientSumSpec | None = None
  response: BernoulliResponseSpec | None = None
  feature_columns: tuple[str, ...] = field(default_factory=tuple)
  bin_map: BinMapSpec | None = None


@dataclass(frozen=True)
class GenerationSpec:
  """Top-level specification for config-driven synthetic generation."""

  name: str
  source_path: Path
  source_format: SourceFormat
  schema: DatasetSchema
  steps: tuple[GenerationStep, ...]
  model_backend: ModelBackendSpec = field(default_factory=ModelBackendSpec)
  n_samples: int = 100_000
  random_seed: int = 343
  source_member: str | None = None


@dataclass(frozen=True)
class EmpiricalStepArtifact:
  """Persisted empirical distribution for a fitted step."""

  step: GenerationStep
  values: tuple[BackendScalarValue, ...]
  probabilities: tuple[float, ...]


@dataclass(frozen=True)
class TruncatedNormalStepArtifact:
  """Persisted numeric summary statistics for a fitted step."""

  step: GenerationStep
  observed_mean: float
  observed_std: float
  minimum: float
  maximum: float
  round_to_int: bool


@dataclass(frozen=True)
class CategoricalModelStepArtifact:
  """Persisted model state for a fitted categorical-model step."""

  step: GenerationStep
  backend_kind: str
  input_columns: tuple[str, ...]
  feature_order: tuple[str, ...]
  learned_categories: tuple[str, ...]
  fitted_model: Any


@dataclass(frozen=True)
class ConfiguredStepArtifact:
  """Persisted configuration-only step with no source-side fitted state."""

  step: GenerationStep


FittedStepArtifact: TypeAlias = (
  EmpiricalStepArtifact
  | TruncatedNormalStepArtifact
  | CategoricalModelStepArtifact
  | ConfiguredStepArtifact
)


@dataclass(frozen=True)
class FittedSynthesizer:
  """Fitted synthesizer bundle that can generate without source data."""

  spec: GenerationSpec
  step_artifacts: tuple[FittedStepArtifact, ...]
  package_version: str
  bundle_format_version: int = 2


@dataclass(frozen=True)
class MetricResult:
  """One quality or disclosure-risk metric in an evaluation report."""

  category: str
  name: str
  column: str | None = None
  values: dict[str, MetricScalar] = field(default_factory=dict)
  details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
  """Evaluation report comparing source and synthetic tabular data."""

  source_row_count: int
  synthetic_row_count: int
  evaluated_columns: tuple[str, ...]
  skipped_columns: tuple[str, ...] = field(default_factory=tuple)
  quality_metrics: tuple[MetricResult, ...] = field(default_factory=tuple)
  disclosure_risk_metrics: tuple[MetricResult, ...] = field(default_factory=tuple)
