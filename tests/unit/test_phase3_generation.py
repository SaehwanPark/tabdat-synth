from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tabdat_synth.core.validation import validate_generation_spec
from tabdat_synth.domain.contracts import (
  ColumnSpec,
  DatasetSchema,
  GenerationSpec,
  GenerationStep,
)
from tabdat_synth.interfaces.io import (
  load_generation_spec,
  load_generation_spec_mapping,
  load_source_dataframe,
)


def _example_spec_path() -> Path:
  return Path(__file__).resolve().parents[2] / "docs/examples/de_synpuf_beneficiary_phase3.yaml"


def test_load_generation_spec_parses_yaml_example():
  spec = load_generation_spec(_example_spec_path())

  assert spec.name == "de_synpuf_beneficiary_phase3"
  assert spec.source_format == "zip_csv"
  assert spec.n_samples == 12
  assert spec.model_backend.kind == "lightgbm"
  assert spec.model_backend.params == {"n_estimators": 8, "verbosity": -1}
  assert spec.steps[0].output_column == "BENE_ESRD_IND"
  assert spec.steps[0].parent_columns == (
    "SP_DIABETES",
    "SP_CHF",
    "BENE_HMO_CVRAGE_TOT_MONS",
  )
  assert "BENE_HMO_CVRAGE_TOT_MONS" in spec.schema.columns_by_name


def test_load_source_dataframe_supports_csv_zip_csv_and_parquet(tmp_path):
  spec = load_generation_spec(_example_spec_path())

  csv_path = spec.source_path.with_suffix(".csv")
  csv_df = load_source_dataframe(csv_path, "csv")
  assert csv_df.height == 9

  zip_df = load_source_dataframe(spec.source_path, spec.source_format, spec.source_member)
  assert zip_df.shape == csv_df.shape

  parquet_path = tmp_path / "tiny.parquet"
  csv_df.write_parquet(parquet_path)
  parquet_df = load_source_dataframe(parquet_path, "parquet")
  assert parquet_df.shape == csv_df.shape


def test_validate_generation_spec_rejects_unknown_parent():
  spec = load_generation_spec(_example_spec_path())
  broken_steps = (
    replace(spec.steps[0], parent_columns=("DOES_NOT_EXIST",)),
    *spec.steps[1:],
  )
  broken_spec = replace(spec, steps=broken_steps)

  with pytest.raises(ValueError, match="unknown dependency column 'DOES_NOT_EXIST'"):
    validate_generation_spec(broken_spec)


def test_validate_generation_spec_rejects_cycles():
  schema = DatasetSchema(
    columns=(
      ColumnSpec(
        name="A",
        data_type="categorical",
        allowed_values=("0", "1"),
      ),
      ColumnSpec(
        name="B",
        data_type="categorical",
        allowed_values=("0", "1"),
      ),
    )
  )
  spec = GenerationSpec(
    name="cycle",
    source_path=Path("cycle.csv"),
    source_format="csv",
    schema=schema,
    steps=(
      GenerationStep(
        name="a_from_b",
        kind="categorical_model",
        output_column="A",
        source_column="A",
        parent_columns=("B",),
      ),
      GenerationStep(
        name="b_from_a",
        kind="categorical_model",
        output_column="B",
        source_column="B",
        parent_columns=("A",),
      ),
    ),
    n_samples=4,
  )

  with pytest.raises(ValueError, match="contains a cycle"):
    validate_generation_spec(spec)


def test_validate_generation_spec_rejects_invalid_missing_value_policy():
  schema = DatasetSchema(
    columns=(
      ColumnSpec(
        name="score",
        data_type="integer",
        missing_value_policy="as_category",
      ),
    )
  )
  spec = GenerationSpec(
    name="bad_policy",
    source_path=Path("score.csv"),
    source_format="csv",
    schema=schema,
    steps=(
      GenerationStep(
        name="score_empirical",
        kind="empirical",
        output_column="score",
        source_column="score",
      ),
    ),
    n_samples=4,
  )

  with pytest.raises(ValueError, match="invalid missing_value_policy"):
    validate_generation_spec(spec)


def test_load_generation_spec_mapping_normalizes_legacy_lightgbm_fields():
  spec = load_generation_spec_mapping(
    {
      "name": "legacy_lightgbm",
      "source": {"path": "tests/fixtures/de_synpuf_beneficiary_tiny.csv", "format": "csv"},
      "lgb_n_estimators": 5,
      "lgb_verbosity": -1,
      "schema": {
        "columns": [
          {"name": "BENE_SEX_IDENT_CD", "data_type": "categorical", "allowed_values": ["1", "2"]},
        ]
      },
      "steps": [
        {"name": "sex_empirical", "kind": "empirical", "output_column": "BENE_SEX_IDENT_CD"},
      ],
    }
  )

  assert spec.model_backend.kind == "lightgbm"
  assert spec.model_backend.params == {"n_estimators": 5, "verbosity": -1}


def test_load_generation_spec_mapping_prefers_explicit_model_backend():
  spec = load_generation_spec_mapping(
    {
      "name": "explicit_backend",
      "source": {"path": "tests/fixtures/de_synpuf_beneficiary_tiny.csv", "format": "csv"},
      "lgb_n_estimators": 3,
      "model_backend": {
        "kind": "logistic_regression",
        "params": {"max_iter": 250, "C": 0.5},
      },
      "schema": {
        "columns": [
          {"name": "BENE_SEX_IDENT_CD", "data_type": "categorical", "allowed_values": ["1", "2"]},
        ]
      },
      "steps": [
        {"name": "sex_empirical", "kind": "empirical", "output_column": "BENE_SEX_IDENT_CD"},
      ],
    }
  )

  assert spec.model_backend.kind == "logistic_regression"
  assert spec.model_backend.params == {"max_iter": 250, "C": 0.5}
