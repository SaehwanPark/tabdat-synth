"""Synthetic data generator package."""

from tabdat_synth.version import __version__
from tabdat_synth.domain.contracts import (
  BernoulliResponseSpec,
  CategoricalModelStepArtifact,
  ColumnSpec,
  CoefficientSumSpec,
  CoefficientTerm,
  ConfiguredStepArtifact,
  EmpiricalStepArtifact,
  DatasetSchema,
  EvaluationReport,
  FittedSynthesizer,
  GenerationSpec,
  GenerationStep,
  MetricResult,
  ModelBackendSpec,
  TruncatedNormalStepArtifact,
)
from tabdat_synth.interfaces.bundles import (
  load_fitted_synthesizer,
  save_fitted_synthesizer,
)
from tabdat_synth.interfaces.io import load_generation_spec
from tabdat_synth.interfaces.reports import (
  evaluation_report_to_dict,
  save_evaluation_report,
)
from tabdat_synth.workflows.evaluation import (
  evaluate_from_spec,
  evaluate_synthetic_data,
)
from tabdat_synth.workflows.pipelines import (
  fit_synthesizer,
  generate_from_fitted,
  generate_from_spec,
)

__all__ = [
  "__version__",
  "BernoulliResponseSpec",
  "CategoricalModelStepArtifact",
  "ColumnSpec",
  "CoefficientSumSpec",
  "CoefficientTerm",
  "ConfiguredStepArtifact",
  "DatasetSchema",
  "EmpiricalStepArtifact",
  "EvaluationReport",
  "MetricResult",
  "FittedSynthesizer",
  "GenerationSpec",
  "GenerationStep",
  "ModelBackendSpec",
  "TruncatedNormalStepArtifact",
  "evaluate_from_spec",
  "evaluate_synthetic_data",
  "evaluation_report_to_dict",
  "fit_synthesizer",
  "generate_from_fitted",
  "generate_from_spec",
  "load_fitted_synthesizer",
  "load_generation_spec",
  "save_evaluation_report",
  "save_fitted_synthesizer",
]
