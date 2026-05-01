from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from tabdat_synth import (
  fit_synthesizer,
  generate_from_fitted,
  load_fitted_synthesizer,
  load_generation_spec,
  save_fitted_synthesizer,
)
from tabdat_synth.domain.contracts import ModelBackendSpec
from tabdat_synth.interfaces.models import (
  register_model_backend,
  unregister_model_backend,
)


def _example_spec_path() -> Path:
  return Path(__file__).resolve().parents[2] / "docs/examples/de_synpuf_beneficiary_phase3.yaml"


def _coefficient_example_spec_path() -> Path:
  return (
    Path(__file__).resolve().parents[2]
    / "docs/examples/tiny_coefficient_outcomes.yaml"
  )


def _example_spec() -> object:
  return replace(
    load_generation_spec(_example_spec_path()),
    n_samples=8,
    model_backend=ModelBackendSpec(
      kind="lightgbm",
      params={"n_estimators": 1, "verbosity": -1},
    ),
  )


def _coefficient_example_spec() -> object:
  return replace(load_generation_spec(_coefficient_example_spec_path()), n_samples=8)


def test_fit_save_load_generate_roundtrip_is_deterministic(tmp_path):
  fitted = fit_synthesizer(_example_spec())
  bundle_path = tmp_path / "bundle"
  save_fitted_synthesizer(fitted, bundle_path)

  metadata = json.loads((bundle_path / "metadata.json").read_text())
  assert metadata["artifact_counts"] == {
    "categorical_model": 3,
    "empirical": 2,
    "truncated_normal": 4,
  }
  assert metadata["model_backend"]["kind"] == "lightgbm"
  assert metadata["source"] == {
    "format": "zip_csv",
    "member": "de_synpuf_beneficiary_tiny.csv",
  }
  assert len(metadata["spec_hash"]) == 64
  assert len(metadata["schema_hash"]) == 64
  assert metadata["python"]["implementation"]
  assert metadata["python"]["version"]
  assert metadata["platform"]["system"]
  assert "polars" in metadata["dependency_versions"]
  assert "scikit-learn" in metadata["dependency_versions"]

  loaded = load_fitted_synthesizer(bundle_path)
  first = generate_from_fitted(loaded)
  second = generate_from_fitted(loaded)

  assert first.equals(second)
  assert first.height == 8


def test_fit_save_load_generate_roundtrip_supports_configured_step_kinds(tmp_path):
  fitted = fit_synthesizer(_coefficient_example_spec())
  bundle_path = tmp_path / "bundle"
  save_fitted_synthesizer(fitted, bundle_path)

  loaded = load_fitted_synthesizer(bundle_path)
  generated = generate_from_fitted(loaded)

  assert generated.height == 8
  assert generated.select(pl.col("pEligible").is_between(0, 1).all()).item()
  assert set(generated["Eligible"].to_list()) <= {0, 1}


def test_loaded_bundle_generates_without_source_data_reads(tmp_path):
  fitted = fit_synthesizer(_example_spec())
  bundle_path = tmp_path / "bundle"
  save_fitted_synthesizer(fitted, bundle_path)

  loaded = load_fitted_synthesizer(bundle_path)
  broken_source = replace(loaded.spec, source_path=Path("/path/does/not/exist.csv"))
  detached = replace(loaded, spec=broken_source)

  result = generate_from_fitted(detached)
  assert result.height == 8
  assert set(result.columns)


def test_generate_from_fitted_allows_n_samples_and_seed_overrides():
  fitted = fit_synthesizer(_example_spec())

  default = generate_from_fitted(fitted)
  resized = generate_from_fitted(fitted, n_samples=5, random_seed=999)

  assert default.height == 8
  assert resized.height == 5
  assert default.columns == resized.columns
  assert not default.head(5).equals(resized)


def test_load_fitted_bundle_rejects_version_mismatch(tmp_path):
  fitted = fit_synthesizer(_example_spec())
  bundle_path = tmp_path / "bundle"
  save_fitted_synthesizer(fitted, bundle_path)

  metadata_path = bundle_path / "metadata.json"
  metadata = json.loads(metadata_path.read_text())
  metadata["package_version"] = "999.0.0"
  metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

  with pytest.raises(ValueError, match="package version mismatch"):
    load_fitted_synthesizer(bundle_path)


def test_load_fitted_bundle_rejects_missing_model_file(tmp_path):
  fitted = fit_synthesizer(_example_spec())
  bundle_path = tmp_path / "bundle"
  save_fitted_synthesizer(fitted, bundle_path)

  model_files = sorted((bundle_path / "steps").glob("*.pkl"))
  assert model_files
  model_files[0].unlink()

  with pytest.raises(ValueError, match="missing required model file"):
    load_fitted_synthesizer(bundle_path)


class _DeterministicBackendModel:
  def __init__(self, classes: list[str]):
    self.classes_ = np.asarray(classes)

  def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
    probabilities[:, 0] = 0.4
    probabilities[:, 1] = 0.6
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


def test_load_fitted_bundle_rejects_unregistered_backend(tmp_path):
  backend_name = "regression_deterministic_backend"
  register_model_backend(backend_name, _fit_deterministic_backend)
  spec = replace(
    load_generation_spec(_example_spec_path()),
    n_samples=8,
    model_backend=ModelBackendSpec(kind=backend_name),
  )

  try:
    fitted = fit_synthesizer(spec)
    bundle_path = tmp_path / "bundle"
    save_fitted_synthesizer(fitted, bundle_path)
  finally:
    unregister_model_backend(backend_name)

  with pytest.raises(ValueError, match="is not registered"):
    load_fitted_synthesizer(bundle_path)
