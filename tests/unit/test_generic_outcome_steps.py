from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tabdat_synth import (
  fit_synthesizer,
  generate_from_fitted,
  generate_from_spec,
)
from tabdat_synth.core.validation import validate_generation_spec
from tabdat_synth.domain.contracts import (
  BernoulliResponseSpec,
  CategoricalModelStepArtifact,
  CoefficientSumSpec,
  CoefficientTerm,
  ColumnSpec,
  DatasetSchema,
  GenerationSpec,
  GenerationStep,
  ModelBackendSpec,
)
from tabdat_synth.interfaces.io import load_generation_spec_mapping


def _write_csv(path: Path, lines: list[str]) -> Path:
  path.write_text("\n".join(lines) + "\n")
  return path


def test_categorical_model_artifact_uses_incoming_ancestor_closure(tmp_path):
  csv_path = _write_csv(
    tmp_path / "closure.csv",
    [
      "A,B,C",
      "0,no,left",
      "0,no,left",
      "1,yes,right",
      "1,yes,right",
    ],
  )
  spec = GenerationSpec(
    name="ancestor_closure",
    source_path=csv_path,
    source_format="csv",
    schema=DatasetSchema(
      columns=(
        ColumnSpec(name="A", data_type="categorical", allowed_values=("0", "1")),
        ColumnSpec(name="B", data_type="categorical", allowed_values=("no", "yes")),
        ColumnSpec(
          name="C",
          data_type="categorical",
          allowed_values=("left", "right"),
        ),
      )
    ),
    steps=(
      GenerationStep(name="a", kind="empirical", output_column="A", source_column="A"),
      GenerationStep(
        name="b",
        kind="categorical_model",
        output_column="B",
        source_column="B",
        parent_columns=("A",),
      ),
      GenerationStep(
        name="c",
        kind="categorical_model",
        output_column="C",
        source_column="C",
        parent_columns=("B",),
      ),
    ),
    model_backend=ModelBackendSpec(
      kind="logistic_regression",
      params={"solver": "liblinear", "max_iter": 500},
    ),
    n_samples=4,
    random_seed=343,
  )

  fitted = fit_synthesizer(spec)
  artifacts = [
    artifact
    for artifact in fitted.step_artifacts
    if isinstance(artifact, CategoricalModelStepArtifact)
  ]

  assert artifacts[0].input_columns == ("A",)
  assert artifacts[1].input_columns == ("A", "B")


def test_generate_from_spec_supports_generic_outcome_steps(tmp_path):
  csv_path = _write_csv(
    tmp_path / "generic_outcomes.csv",
    [
      "sex,score",
      "1,1.0",
      "2,2.0",
      "1,3.0",
      "2,4.0",
    ],
  )
  spec = GenerationSpec(
    name="generic_outcomes",
    source_path=csv_path,
    source_format="csv",
    schema=DatasetSchema(
      columns=(
        ColumnSpec(name="sex", data_type="categorical", allowed_values=("1", "2")),
        ColumnSpec(name="score", data_type="float", minimum=0.0, maximum=5.0),
        ColumnSpec(name="risk_log_odds", data_type="float"),
        ColumnSpec(name="risk_prob", data_type="float"),
        ColumnSpec(name="response_flag", data_type="integer"),
      )
    ),
    steps=(
      GenerationStep(
        name="sex",
        kind="empirical",
        output_column="sex",
        source_column="sex",
      ),
      GenerationStep(
        name="score",
        kind="truncated_normal",
        output_column="score",
        source_column="score",
      ),
      GenerationStep(
        name="risk_log_odds",
        kind="coefficient_sum",
        output_column="risk_log_odds",
        parent_columns=("sex", "score"),
        coefficient_sum=CoefficientSumSpec(
          intercept=-1.0,
          terms=(
            CoefficientTerm(feature_name="sex", coefficient=0.75, value="2"),
            CoefficientTerm(feature_name="score", coefficient=0.5),
          ),
        ),
      ),
      GenerationStep(
        name="risk_prob",
        kind="sigmoid_probability",
        output_column="risk_prob",
        input_column="risk_log_odds",
      ),
      GenerationStep(
        name="response_flag",
        kind="bernoulli_response",
        output_column="response_flag",
        input_column="risk_log_odds",
        response=BernoulliResponseSpec(intercept=0.2, slope=1.1),
      ),
    ),
    n_samples=8,
    random_seed=343,
  )

  first = generate_from_spec(spec)
  second = generate_from_spec(spec)
  fitted = fit_synthesizer(spec)
  same_seed = generate_from_fitted(fitted, random_seed=999)
  same_seed_again = generate_from_fitted(fitted, random_seed=999)
  different_seed = generate_from_fitted(fitted, random_seed=1000)

  assert first.equals(second)
  assert first.height == 8
  assert first["risk_prob"].to_numpy().tolist() == pytest.approx(
    (1.0 / (1.0 + np.exp(-first["risk_log_odds"].to_numpy()))).tolist()
  )
  assert set(first["response_flag"].to_list()) <= {0, 1}
  assert same_seed.equals(same_seed_again)
  assert same_seed.columns == different_seed.columns
  assert same_seed.dtypes == different_seed.dtypes
  assert not same_seed.equals(different_seed)
  assert same_seed.select(pl.col("risk_prob").is_between(0, 1).all()).item()
  assert set(same_seed["response_flag"].to_list()) <= {0, 1}


def test_validate_generation_spec_rejects_non_terminal_bernoulli_outputs():
  spec = GenerationSpec(
    name="bad_bernoulli_dependency",
    source_path=Path("tiny.csv"),
    source_format="csv",
    schema=DatasetSchema(
      columns=(
        ColumnSpec(name="score", data_type="float"),
        ColumnSpec(name="response_flag", data_type="integer"),
        ColumnSpec(name="downstream", data_type="float"),
      )
    ),
    steps=(
      GenerationStep(
        name="score",
        kind="empirical",
        output_column="score",
        source_column="score",
      ),
      GenerationStep(
        name="response_flag",
        kind="bernoulli_response",
        output_column="response_flag",
        input_column="score",
        response=BernoulliResponseSpec(),
      ),
      GenerationStep(
        name="downstream",
        kind="coefficient_sum",
        output_column="downstream",
        parent_columns=("response_flag",),
        coefficient_sum=CoefficientSumSpec(
          terms=(CoefficientTerm(feature_name="response_flag", coefficient=1.0),)
        ),
      ),
    ),
  )

  with pytest.raises(ValueError, match="bernoulli_response outputs must be terminal"):
    validate_generation_spec(spec)


def test_load_generation_spec_mapping_rejects_removed_va_outcomes():
  with pytest.raises(ValueError, match="va_outcomes"):
    load_generation_spec_mapping(
      {
        "name": "no_va_outcomes",
        "source": {"path": "tests/fixtures/de_synpuf_beneficiary_tiny.csv", "format": "csv"},
        "schema": {
          "columns": [
            {"name": "score", "data_type": "float"},
          ]
        },
        "steps": [
          {"name": "score", "kind": "empirical", "output_column": "score"},
        ],
        "va_outcomes": {"kind": "legacy_va_outcomes"},
      }
    )
