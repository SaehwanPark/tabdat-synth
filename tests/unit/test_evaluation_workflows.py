from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from tabdat_synth import (
  evaluate_from_spec,
  evaluate_synthetic_data,
  evaluation_report_to_dict,
  generate_from_spec,
  load_generation_spec,
  save_evaluation_report,
)
from tabdat_synth.domain.contracts import (
  ColumnSpec,
  DatasetSchema,
)


def _coefficient_example_spec_path() -> Path:
  return (
    Path(__file__).resolve().parents[2]
    / "docs/examples/tiny_coefficient_outcomes.yaml"
  )


def test_evaluate_synthetic_data_reports_quality_risk_and_skipped_columns():
  schema = DatasetSchema(
    columns=(
      ColumnSpec(name="patient_id", data_type="integer", role="identifier"),
      ColumnSpec(name="category", data_type="categorical"),
      ColumnSpec(name="score", data_type="float"),
      ColumnSpec(name="generated_outcome", data_type="float", role="outcome"),
    )
  )
  source = pl.DataFrame(
    {
      "patient_id": [1, 2],
      "category": ["a", "b"],
      "score": [1.0, 2.0],
    }
  )
  synthetic = pl.DataFrame(
    {
      "patient_id": [99, 100],
      "category": ["a", "b"],
      "score": [1.5, 2.0],
      "generated_outcome": [0.1, 0.2],
    }
  )

  report = evaluate_synthetic_data(source, synthetic, schema)
  metric_names = {metric.name for metric in report.quality_metrics}
  risk_names = {metric.name for metric in report.disclosure_risk_metrics}

  assert report.source_row_count == 2
  assert report.synthetic_row_count == 2
  assert report.evaluated_columns == ("category", "score")
  assert report.skipped_columns == ("generated_outcome",)
  assert "categorical_total_variation_distance" in metric_names
  assert "numeric_ks_statistic" in metric_names
  assert risk_names == {
    "exact_row_match_rate",
    "nearest_neighbor_distance_summary",
    "source_duplicate_row_rate",
    "synthetic_duplicate_row_rate",
  }


def test_evaluate_from_spec_loads_source_and_skips_generated_only_outputs():
  spec = load_generation_spec(_coefficient_example_spec_path())
  synthetic = generate_from_spec(spec)

  report = evaluate_from_spec(spec, synthetic, max_pairwise_rows=4)

  assert report.source_row_count == 9
  assert report.synthetic_row_count == spec.n_samples
  assert report.evaluated_columns == (
    "BENE_SEX_IDENT_CD",
    "BENE_RACE_CD",
    "PLAN_CVRG_MOS_NUM",
  )
  assert report.skipped_columns == (
    "eligibility_log_odds",
    "pEligible",
    "Eligible",
  )


def test_evaluation_report_serializes_to_dict_and_json(tmp_path):
  schema = DatasetSchema(
    columns=(
      ColumnSpec(name="category", data_type="categorical"),
      ColumnSpec(name="score", data_type="float"),
    )
  )
  source = pl.DataFrame({"category": ["a", "b"], "score": [1.0, 2.0]})
  synthetic = pl.DataFrame({"category": ["a", "c"], "score": [1.0, 3.0]})
  report = evaluate_synthetic_data(source, synthetic, schema)

  payload = evaluation_report_to_dict(report)
  report_path = save_evaluation_report(report, tmp_path / "evaluation.json")
  saved = json.loads(report_path.read_text())

  assert payload["evaluated_columns"] == ["category", "score"]
  assert saved == payload
  assert report_path == (tmp_path / "evaluation.json").resolve()
