from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import pickle
import platform
import re

import yaml  # type: ignore[import-untyped]

from tabdat_synth.domain.contracts import (
  BernoulliResponseSpec,
  CategoricalModelStepArtifact,
  CoefficientSumSpec,
  CoefficientTerm,
  ConfiguredStepArtifact,
  EmpiricalStepArtifact,
  FittedStepArtifact,
  FittedSynthesizer,
  GenerationStep,
  TruncatedNormalStepArtifact,
)
from tabdat_synth.interfaces.io import (
  generation_spec_to_dict,
  load_generation_spec_mapping,
)
from tabdat_synth.interfaces.models import require_model_backend
from tabdat_synth.version import __version__

BUNDLE_FORMAT_VERSION = 2
_METADATA_FILENAME = "metadata.json"
_SPEC_FILENAME = "spec.yaml"
_STEPS_DIRNAME = "steps"
_SOURCE_REQUIRED_STEP_KINDS = {"empirical", "truncated_normal", "categorical_model"}
_METADATA_DEPENDENCIES = (
  "lightgbm",
  "numpy",
  "pandas",
  "polars",
  "pyarrow",
  "PyYAML",
  "scikit-learn",
  "scipy",
)


def save_fitted_synthesizer(
  fitted: FittedSynthesizer,
  path: Path | str,
) -> Path:
  """Persist a fitted synthesizer bundle directory to disk."""
  bundle_path = Path(path).resolve()
  _prepare_bundle_directory(bundle_path)

  steps_dir = bundle_path / _STEPS_DIRNAME
  steps_dir.mkdir()

  metadata = _build_bundle_metadata(fitted)
  (bundle_path / _METADATA_FILENAME).write_text(
    json.dumps(metadata, indent=2, sort_keys=True)
  )
  (bundle_path / _SPEC_FILENAME).write_text(
    yaml.safe_dump(generation_spec_to_dict(fitted.spec), sort_keys=False)
  )

  for index, artifact in enumerate(fitted.step_artifacts):
    _write_step_artifact(steps_dir, index, artifact)

  return bundle_path


def _build_bundle_metadata(fitted: FittedSynthesizer) -> dict[str, object]:
  spec_payload = generation_spec_to_dict(fitted.spec)
  return {
    "bundle_format_version": fitted.bundle_format_version,
    "package_version": fitted.package_version,
    "created_at": datetime.now(UTC).isoformat(),
    "step_count": len(fitted.step_artifacts),
    "artifact_counts": _artifact_counts(fitted.step_artifacts),
    "dependency_versions": _dependency_versions(),
    "model_backend": {
      "kind": fitted.spec.model_backend.kind,
      "params": dict(fitted.spec.model_backend.params),
    },
    "platform": {
      "machine": platform.machine(),
      "platform": platform.platform(),
      "release": platform.release(),
      "system": platform.system(),
    },
    "python": {
      "implementation": platform.python_implementation(),
      "version": platform.python_version(),
    },
    "schema_hash": _stable_payload_hash(spec_payload["schema"]),
    "source": {
      "format": fitted.spec.source_format,
      "member": fitted.spec.source_member,
    },
    "spec_hash": _stable_payload_hash(spec_payload),
  }


def _dependency_versions() -> dict[str, str | None]:
  versions: dict[str, str | None] = {}
  for package_name in _METADATA_DEPENDENCIES:
    try:
      versions[package_name] = version(package_name)
    except PackageNotFoundError:
      versions[package_name] = None
  return versions


def _stable_payload_hash(payload: object) -> str:
  encoded = json.dumps(
    payload,
    default=str,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _artifact_counts(artifacts: tuple[FittedStepArtifact, ...]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for artifact in artifacts:
    artifact_kind = _artifact_kind(artifact)
    counts[artifact_kind] = counts.get(artifact_kind, 0) + 1
  return counts


def _artifact_kind(artifact: FittedStepArtifact) -> str:
  if isinstance(artifact, EmpiricalStepArtifact):
    return "empirical"
  if isinstance(artifact, TruncatedNormalStepArtifact):
    return "truncated_normal"
  if isinstance(artifact, CategoricalModelStepArtifact):
    return "categorical_model"
  if isinstance(artifact, ConfiguredStepArtifact):
    return "configured"
  raise ValueError(f"unsupported fitted step artifact type: {type(artifact)}")


def load_fitted_synthesizer(path: Path | str) -> FittedSynthesizer:
  """Load a persisted fitted synthesizer bundle from disk."""
  bundle_path = Path(path).resolve()
  metadata_path = bundle_path / _METADATA_FILENAME
  spec_path = bundle_path / _SPEC_FILENAME
  steps_dir = bundle_path / _STEPS_DIRNAME

  for required_path in (metadata_path, spec_path, steps_dir):
    if not required_path.exists():
      raise ValueError(f"fitted bundle is missing required path: {required_path}")

  metadata = json.loads(metadata_path.read_text())
  bundle_format_version = int(metadata["bundle_format_version"])
  if bundle_format_version != BUNDLE_FORMAT_VERSION:
    raise ValueError(
      "fitted bundle format version mismatch: "
      f"expected {BUNDLE_FORMAT_VERSION}, got {bundle_format_version}"
    )

  package_version = str(metadata["package_version"])
  if package_version != __version__:
    raise ValueError(
      "fitted bundle package version mismatch: "
      f"expected {__version__}, got {package_version}"
    )

  raw_spec = yaml.safe_load(spec_path.read_text())
  spec = load_generation_spec_mapping(raw_spec)
  step_payload_paths = sorted(steps_dir.glob("*.json"))
  expected_step_count = int(metadata["step_count"])
  if len(step_payload_paths) != expected_step_count:
    raise ValueError(
      "fitted bundle step metadata count mismatch: "
      f"expected {expected_step_count}, found {len(step_payload_paths)}"
    )

  step_artifacts = tuple(
    _read_step_artifact(steps_dir, payload_path)
    for payload_path in step_payload_paths
  )
  return FittedSynthesizer(
    spec=spec,
    step_artifacts=step_artifacts,
    package_version=package_version,
    bundle_format_version=bundle_format_version,
  )


def _prepare_bundle_directory(bundle_path: Path) -> None:
  if bundle_path.exists():
    if not bundle_path.is_dir():
      raise ValueError(f"bundle path is not a directory: {bundle_path}")
    if any(bundle_path.iterdir()):
      raise ValueError(f"bundle path must be empty before saving: {bundle_path}")
    return

  bundle_path.mkdir(parents=True)


def _write_step_artifact(
  steps_dir: Path,
  index: int,
  artifact: FittedStepArtifact,
) -> None:
  stem = f"{index:03d}_{_safe_token(artifact.step.output_column)}"
  payload_path = steps_dir / f"{stem}.json"

  if isinstance(artifact, EmpiricalStepArtifact):
    empirical_payload: dict[str, object] = {
      "artifact_kind": "empirical",
      "step": _generation_step_to_dict(artifact.step),
      "values": list(artifact.values),
      "probabilities": list(artifact.probabilities),
    }
    payload_path.write_text(json.dumps(empirical_payload, indent=2, sort_keys=True))
    return

  if isinstance(artifact, TruncatedNormalStepArtifact):
    truncated_payload: dict[str, object] = {
      "artifact_kind": "truncated_normal",
      "step": _generation_step_to_dict(artifact.step),
      "observed_mean": artifact.observed_mean,
      "observed_std": artifact.observed_std,
      "minimum": artifact.minimum,
      "maximum": artifact.maximum,
      "round_to_int": artifact.round_to_int,
    }
    payload_path.write_text(json.dumps(truncated_payload, indent=2, sort_keys=True))
    return

  if isinstance(artifact, CategoricalModelStepArtifact):
    model_filename = f"{stem}.pkl"
    model_payload: dict[str, object] = {
      "artifact_kind": "categorical_model",
      "step": _generation_step_to_dict(artifact.step),
      "backend_kind": artifact.backend_kind,
      "input_columns": list(artifact.input_columns),
      "feature_order": list(artifact.feature_order),
      "learned_categories": list(artifact.learned_categories),
      "model_filename": model_filename,
    }
    payload_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True))
    with (steps_dir / model_filename).open("wb") as handle:
      pickle.dump(artifact.fitted_model, handle)
    return

  if not isinstance(artifact, ConfiguredStepArtifact):
    raise ValueError(
      f"unsupported step artifact type for bundle persistence: {type(artifact)}"
    )

  configured_payload: dict[str, object] = {
    "artifact_kind": "configured",
    "step": _generation_step_to_dict(artifact.step),
  }
  payload_path.write_text(json.dumps(configured_payload, indent=2, sort_keys=True))


def _read_step_artifact(steps_dir: Path, payload_path: Path) -> FittedStepArtifact:
  payload = json.loads(payload_path.read_text())
  artifact_kind = payload["artifact_kind"]
  step = _load_generation_step(payload["step"])

  if artifact_kind == "empirical":
    return EmpiricalStepArtifact(
      step=step,
      values=tuple(payload["values"]),
      probabilities=tuple(float(value) for value in payload["probabilities"]),
    )

  if artifact_kind == "truncated_normal":
    return TruncatedNormalStepArtifact(
      step=step,
      observed_mean=float(payload["observed_mean"]),
      observed_std=float(payload["observed_std"]),
      minimum=float(payload["minimum"]),
      maximum=float(payload["maximum"]),
      round_to_int=bool(payload["round_to_int"]),
    )

  if artifact_kind == "configured":
    return ConfiguredStepArtifact(step=step)

  if artifact_kind != "categorical_model":
    raise ValueError(
      f"fitted bundle contains unsupported step artifact kind: {artifact_kind}"
    )

  backend_kind = str(payload["backend_kind"])
  require_model_backend(backend_kind)
  model_path = steps_dir / str(payload["model_filename"])
  if not model_path.exists():
    raise ValueError(f"fitted bundle is missing required model file: {model_path}")

  with model_path.open("rb") as handle:
    fitted_model = pickle.load(handle)

  return CategoricalModelStepArtifact(
    step=step,
    backend_kind=backend_kind,
    input_columns=tuple(str(value) for value in payload["input_columns"]),
    feature_order=tuple(str(value) for value in payload["feature_order"]),
    learned_categories=tuple(str(value) for value in payload["learned_categories"]),
    fitted_model=fitted_model,
  )


def _load_generation_step(raw_step: object) -> GenerationStep:
  if not isinstance(raw_step, dict):
    raise ValueError("bundle step payload must be a mapping")

  kind = raw_step["kind"]
  source_column = raw_step.get("source_column")
  if source_column is None and kind in _SOURCE_REQUIRED_STEP_KINDS:
    source_column = raw_step["output_column"]

  return GenerationStep(
    name=raw_step["name"],
    kind=kind,
    output_column=raw_step["output_column"],
    source_column=source_column,
    parent_columns=tuple(raw_step.get("parent_columns", [])),
    input_column=raw_step.get("input_column"),
    mean_shift=float(raw_step.get("mean_shift", 0.0)),
    coefficient_sum=_load_coefficient_sum(raw_step.get("coefficient_sum")),
    response=_load_response(raw_step.get("response")),
  )


def _generation_step_to_dict(step: GenerationStep) -> dict[str, object]:
  payload: dict[str, object] = {
    "name": step.name,
    "kind": step.kind,
    "output_column": step.output_column,
  }
  if step.source_column is not None and step.source_column != step.output_column:
    payload["source_column"] = step.source_column
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
  return payload


def _load_coefficient_sum(raw_coefficient_sum: object) -> CoefficientSumSpec | None:
  if raw_coefficient_sum is None:
    return None
  if not isinstance(raw_coefficient_sum, dict):
    raise ValueError("bundle step coefficient_sum must be a mapping")

  raw_terms = raw_coefficient_sum.get("terms", [])
  if not isinstance(raw_terms, list):
    raise ValueError("bundle step coefficient_sum.terms must be a list")

  terms = []
  for raw_term in raw_terms:
    if not isinstance(raw_term, dict):
      raise ValueError("bundle step coefficient_sum terms must be mappings")
    terms.append(
      CoefficientTerm(
        feature_name=raw_term["feature_name"],
        coefficient=float(raw_term["coefficient"]),
        value=str(raw_term["value"]) if "value" in raw_term else None,
      )
    )

  return CoefficientSumSpec(
    intercept=float(raw_coefficient_sum.get("intercept", 0.0)),
    reference_value=str(raw_coefficient_sum.get("reference_value", "99")),
    terms=tuple(terms),
  )


def _coefficient_sum_to_dict(coefficient_sum: CoefficientSumSpec) -> dict[str, object]:
  return {
    "intercept": coefficient_sum.intercept,
    "reference_value": coefficient_sum.reference_value,
    "terms": [
      _coefficient_term_to_dict(term) for term in coefficient_sum.terms
    ],
  }


def _coefficient_term_to_dict(term: CoefficientTerm) -> dict[str, object]:
  payload: dict[str, object] = {
    "feature_name": term.feature_name,
    "coefficient": term.coefficient,
  }
  if term.value is not None:
    payload["value"] = term.value
  return payload


def _load_response(raw_response: object) -> BernoulliResponseSpec | None:
  if raw_response is None:
    return None
  if not isinstance(raw_response, dict):
    raise ValueError("bundle step response must be a mapping")
  return BernoulliResponseSpec(
    intercept=float(raw_response.get("intercept", 0.0)),
    slope=float(raw_response.get("slope", 1.0)),
  )


def _response_to_dict(response: BernoulliResponseSpec) -> dict[str, object]:
  return {
    "intercept": response.intercept,
    "slope": response.slope,
  }


def _safe_token(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
