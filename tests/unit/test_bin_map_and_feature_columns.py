from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from tabdat_synth import (
  BinMapSpec,
  fit_synthesizer,
  generate_from_fitted,
  generate_from_spec,
  load_fitted_synthesizer,
  load_generation_spec,
  save_fitted_synthesizer,
)
from tabdat_synth.core.validation import validate_generation_spec
from tabdat_synth.domain.contracts import (
  CategoricalModelStepArtifact,
  ColumnSpec,
  DatasetSchema,
  GenerationSpec,
  GenerationStep,
  ModelBackendSpec,
)

_AGE_BREAKS = (54.0, 64.0, 74.0, 84.0)
_AGE_LABELS = ("1", "2", "3", "4", "5")


def _write_csv(path: Path, lines: list[str]) -> Path:
  path.write_text("\n".join(lines) + "\n")
  return path


def _age_spec(tmp_path: Path, *, feature_columns: tuple[str, ...]) -> GenerationSpec:
  csv_path = _write_csv(
    tmp_path / "ages.csv",
    [
      "age_cont,agegrp,B",
      "40,1,no",
      "40,1,no",
      "55,2,no",
      "62,2,no",
      "70,3,yes",
      "78,3,yes",
      "86,4,yes",
      "90,5,yes",
    ],
  )
  return GenerationSpec(
    name="bin_map_feature_columns",
    source_path=csv_path,
    source_format="csv",
    schema=DatasetSchema(
      columns=(
        ColumnSpec(name="age_cont", data_type="integer"),
        ColumnSpec(
          name="agegrp",
          data_type="categorical",
          allowed_values=_AGE_LABELS,
        ),
        ColumnSpec(name="B", data_type="categorical", allowed_values=("no", "yes")),
      )
    ),
    steps=(
      GenerationStep(
        name="age_cont",
        kind="truncated_normal",
        output_column="age_cont",
        source_column="age_cont",
      ),
      GenerationStep(
        name="agegrp",
        kind="bin_map",
        output_column="agegrp",
        input_column="age_cont",
        bin_map=BinMapSpec(breaks=_AGE_BREAKS, labels=_AGE_LABELS),
      ),
      GenerationStep(
        name="b",
        kind="categorical_model",
        output_column="B",
        source_column="B",
        parent_columns=("agegrp",),
        feature_columns=feature_columns,
      ),
    ),
    model_backend=ModelBackendSpec(
      kind="logistic_regression",
      params={"solver": "liblinear", "max_iter": 500},
    ),
    n_samples=8,
    random_seed=343,
  )


def test_bin_map_maps_integer_ages_with_can_breaks(tmp_path):
  source = _write_csv(
    tmp_path / "ages_only.csv",
    ["age_cont", "21", "54", "64", "74", "84", "85", "99"],
  )
  spec = GenerationSpec(
    name="age_bins",
    source_path=source,
    source_format="csv",
    schema=DatasetSchema(
      columns=(
        ColumnSpec(name="age_cont", data_type="integer"),
        ColumnSpec(
          name="agegrp",
          data_type="categorical",
          allowed_values=_AGE_LABELS,
        ),
      )
    ),
    steps=(
      GenerationStep(
        name="age_cont",
        kind="truncated_normal",
        output_column="age_cont",
        source_column="age_cont",
      ),
      GenerationStep(
        name="agegrp",
        kind="bin_map",
        output_column="agegrp",
        input_column="age_cont",
        bin_map=BinMapSpec(breaks=_AGE_BREAKS, labels=_AGE_LABELS),
      ),
    ),
    n_samples=8,
    random_seed=343,
  )

  frame = pl.DataFrame(
    {
      "age_cont": [21, 54, 64, 74, 84, 85, 99, 60],
    }
  )
  from tabdat_synth.workflows.pipelines import _calculate_bin_map_series

  binned = _calculate_bin_map_series(spec.steps[1], frame).tolist()
  assert binned == ["1", "1", "2", "3", "4", "5", "5", "2"]

  first = generate_from_spec(spec)
  repeated = generate_from_spec(spec)
  other_seed = generate_from_spec(replace(spec, random_seed=999))

  assert first.equals(repeated)
  assert set(first["agegrp"].to_list()) <= set(_AGE_LABELS)
  assert set(other_seed["agegrp"].to_list()) <= set(_AGE_LABELS)

  # The bin map is a deterministic function of the generated age, so every seed
  # must produce the same labels for the same ages.
  expected = _calculate_bin_map_series(spec.steps[1], first).tolist()
  assert first["agegrp"].to_list() == expected
  expected_other = _calculate_bin_map_series(spec.steps[1], other_seed).tolist()
  assert other_seed["agegrp"].to_list() == expected_other


def test_categorical_feature_columns_override_ancestor_closure(tmp_path):
  spec = _age_spec(tmp_path, feature_columns=("agegrp",))

  fitted = fit_synthesizer(spec)
  categorical = [
    artifact
    for artifact in fitted.step_artifacts
    if isinstance(artifact, CategoricalModelStepArtifact)
  ]

  assert len(categorical) == 1
  assert categorical[0].input_columns == ("agegrp",)
  assert "age_cont" not in categorical[0].input_columns

  frame = generate_from_fitted(fitted, n_samples=16)
  assert set(frame["agegrp"].to_list()) <= set(_AGE_LABELS)


def test_empty_feature_columns_keeps_ancestor_closure(tmp_path):
  spec = _age_spec(tmp_path, feature_columns=())

  fitted = fit_synthesizer(spec)
  categorical = [
    artifact
    for artifact in fitted.step_artifacts
    if isinstance(artifact, CategoricalModelStepArtifact)
  ]

  assert categorical[0].input_columns == ("age_cont", "agegrp")


def test_bundle_with_bin_map_generates_after_source_removal(tmp_path):
  spec = _age_spec(tmp_path, feature_columns=("agegrp",))
  fitted = fit_synthesizer(spec)
  bundle_path = tmp_path / "bundle"
  save_fitted_synthesizer(fitted, bundle_path)

  spec.source_path.unlink()

  loaded = load_fitted_synthesizer(bundle_path)
  detached = replace(
    loaded, spec=replace(loaded.spec, source_path=Path("source/not_distributed.parquet"))
  )
  frame = generate_from_fitted(detached, n_samples=12)

  assert frame.height == 12
  assert frame.columns == ["age_cont", "agegrp", "B"]
  assert set(frame["agegrp"].to_list()) <= set(_AGE_LABELS)
  assert loaded.spec.steps[1].bin_map == BinMapSpec(
    breaks=_AGE_BREAKS, labels=_AGE_LABELS
  )


def test_example_spec_round_trips_without_new_fields(tmp_path):
  example_path = (
    Path(__file__).resolve().parents[2]
    / "docs/examples/tiny_coefficient_outcomes.yaml"
  )
  spec = load_generation_spec(example_path)
  assert all(not step.feature_columns for step in spec.steps)
  assert all(step.bin_map is None for step in spec.steps)

  fitted = fit_synthesizer(replace(spec, n_samples=8))
  bundle_path = tmp_path / "bundle"
  save_fitted_synthesizer(fitted, bundle_path)
  loaded = load_fitted_synthesizer(bundle_path)

  assert loaded.spec == replace(spec, n_samples=8)
  assert generate_from_fitted(loaded).height == 8


def test_validate_generation_spec_rejects_bad_bin_map(tmp_path):
  source = _write_csv(tmp_path / "ages_only.csv", ["age_cont", "21", "54", "64"])

  def _spec(bin_map: BinMapSpec) -> GenerationSpec:
    return GenerationSpec(
      name="bad_bin_map",
      source_path=source,
      source_format="csv",
      schema=DatasetSchema(
        columns=(
          ColumnSpec(name="age_cont", data_type="integer"),
          ColumnSpec(name="agegrp", data_type="categorical"),
        )
      ),
      steps=(
        GenerationStep(
          name="age_cont",
          kind="truncated_normal",
          output_column="age_cont",
          source_column="age_cont",
        ),
        GenerationStep(
          name="agegrp",
          kind="bin_map",
          output_column="agegrp",
          input_column="age_cont",
          bin_map=bin_map,
        ),
      ),
    )

  with pytest.raises(ValueError, match="len\\(labels\\) == len\\(breaks\\) \\+ 1"):
    validate_generation_spec(_spec(BinMapSpec(breaks=(54.0, 64.0), labels=("1",))))

  with pytest.raises(ValueError, match="strictly increasing breaks"):
    validate_generation_spec(
      _spec(BinMapSpec(breaks=(64.0, 54.0), labels=("1", "2", "3")))
    )

  with pytest.raises(ValueError, match="requires categorical output"):
    validate_generation_spec(
      GenerationSpec(
        name="bad_bin_map_output",
        source_path=source,
        source_format="csv",
        schema=DatasetSchema(
          columns=(
            ColumnSpec(name="age_cont", data_type="integer"),
            ColumnSpec(name="agegrp", data_type="float"),
          )
        ),
        steps=(
          GenerationStep(
            name="age_cont",
            kind="truncated_normal",
            output_column="age_cont",
            source_column="age_cont",
          ),
          GenerationStep(
            name="agegrp",
            kind="bin_map",
            output_column="agegrp",
            input_column="age_cont",
            bin_map=BinMapSpec(breaks=(54.0,), labels=("1", "2")),
          ),
        ),
      )
    )


def test_validate_generation_spec_rejects_unknown_feature_columns(tmp_path):
  spec = _age_spec(tmp_path, feature_columns=("age_cont",))
  spec = replace(
    spec,
    steps=(
      spec.steps[0],
      spec.steps[1],
      replace(spec.steps[2], feature_columns=("not_a_column",)),
    ),
  )

  with pytest.raises(ValueError, match="feature_columns must already be ancestor"):
    validate_generation_spec(spec)


def test_bin_map_yaml_round_trip(tmp_path):
  from tabdat_synth.interfaces.io import (
    generation_spec_to_dict,
    load_generation_spec_mapping,
  )

  spec = _age_spec(tmp_path, feature_columns=("agegrp",))
  payload = generation_spec_to_dict(spec)
  reloaded = load_generation_spec_mapping(payload, base_path=tmp_path)

  assert reloaded == spec
  assert reloaded.steps[1].bin_map == BinMapSpec(
    breaks=_AGE_BREAKS, labels=_AGE_LABELS
  )
  assert reloaded.steps[2].feature_columns == ("agegrp",)
