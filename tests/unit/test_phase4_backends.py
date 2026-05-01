from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tabdat_synth import fit_synthesizer, generate_from_fitted, load_generation_spec
from tabdat_synth.core.validation import validate_generation_spec
from tabdat_synth.domain.contracts import (
  CategoricalModelStepArtifact,
  ModelBackendSpec,
)
from tabdat_synth.interfaces.models import (
  register_model_backend,
  unregister_model_backend,
)


def _example_spec_path() -> Path:
  return Path(__file__).resolve().parents[2] / "docs/examples/de_synpuf_beneficiary_phase3.yaml"


@pytest.mark.parametrize(
  "model_backend",
  [
    ModelBackendSpec(
      kind="lightgbm",
      params={"n_estimators": 1, "verbosity": -1},
    ),
    ModelBackendSpec(
      kind="logistic_regression",
      params={"max_iter": 500, "solver": "liblinear"},
    ),
  ],
)
def test_fit_synthesizer_supports_built_in_backends(model_backend: ModelBackendSpec):
  spec = replace(
    load_generation_spec(_example_spec_path()),
    n_samples=6,
    model_backend=model_backend,
  )

  fitted = fit_synthesizer(spec)
  categorical_artifacts = [
    artifact
    for artifact in fitted.step_artifacts
    if isinstance(artifact, CategoricalModelStepArtifact)
  ]

  assert categorical_artifacts
  assert all(artifact.backend_kind == model_backend.kind for artifact in categorical_artifacts)
  assert all(artifact.feature_order for artifact in categorical_artifacts)

  generated = generate_from_fitted(fitted)
  assert generated.height == spec.n_samples


def test_validate_generation_spec_rejects_invalid_model_backend_params():
  spec = replace(
    load_generation_spec(_example_spec_path()),
    model_backend=ModelBackendSpec(
      kind="logistic_regression",
      params={"max_iter": 0},
    ),
  )

  with pytest.raises(ValueError, match="max_iter must be positive"):
    validate_generation_spec(spec)


def test_validate_generation_spec_rejects_unsupported_backend_keys():
  spec = replace(
    load_generation_spec(_example_spec_path()),
    model_backend=ModelBackendSpec(
      kind="lightgbm",
      params={"unsupported": 3},
    ),
  )

  with pytest.raises(ValueError, match="unsupported keys"):
    validate_generation_spec(spec)


class _DeterministicBackendModel:
  def __init__(self, classes: list[str]):
    self.classes_ = np.asarray(classes)

  def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
    probabilities[:, 0] = 0.25
    probabilities[:, 1] = 0.75
    return probabilities


def _fit_deterministic_backend(
  X: pl.DataFrame,
  y: pl.Series,
  model_backend: ModelBackendSpec,
  random_seed: int,
) -> _DeterministicBackendModel:
  del model_backend
  del random_seed
  encoded = pd.get_dummies(X.to_pandas(), dtype=float).astype(float)
  model = _DeterministicBackendModel(
    list(dict.fromkeys(y.cast(pl.String).to_list()))
  )
  setattr(model, "_sdg_feature_order", tuple(str(column) for column in encoded.columns))
  return model


def test_custom_backend_registration_and_missing_backend_failure():
  backend_name = "unit_test_deterministic_backend"
  register_model_backend(backend_name, _fit_deterministic_backend)
  spec = replace(
    load_generation_spec(_example_spec_path()),
    n_samples=6,
    model_backend=ModelBackendSpec(kind=backend_name),
  )

  try:
    fitted = fit_synthesizer(spec)
    generated = generate_from_fitted(fitted)
    assert generated.height == spec.n_samples
  finally:
    unregister_model_backend(backend_name)

  with pytest.raises(ValueError, match="is not registered"):
    fit_synthesizer(spec)
