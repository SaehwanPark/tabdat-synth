from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
import json
import math
from pathlib import Path

from tabdat_synth.domain.contracts import (
  EvaluationReport,
  MetricResult,
)


def evaluation_report_to_dict(report: EvaluationReport) -> dict[str, object]:
  """Convert an evaluation report into a JSON-serializable mapping."""
  return {
    "source_row_count": report.source_row_count,
    "synthetic_row_count": report.synthetic_row_count,
    "evaluated_columns": list(report.evaluated_columns),
    "skipped_columns": list(report.skipped_columns),
    "quality_metrics": [
      _metric_result_to_dict(metric) for metric in report.quality_metrics
    ],
    "disclosure_risk_metrics": [
      _metric_result_to_dict(metric)
      for metric in report.disclosure_risk_metrics
    ],
  }


def save_evaluation_report(report: EvaluationReport, path: Path | str) -> Path:
  """Write an evaluation report to JSON."""
  report_path = Path(path).resolve()
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(
    json.dumps(evaluation_report_to_dict(report), indent=2, sort_keys=True) + "\n"
  )
  return report_path


def _metric_result_to_dict(metric: MetricResult) -> dict[str, object]:
  payload: dict[str, object] = {
    "category": metric.category,
    "name": metric.name,
    "values": _jsonable(metric.values),
    "details": _jsonable(metric.details),
  }
  if metric.column is not None:
    payload["column"] = metric.column
  return payload


def _jsonable(value: object) -> object:
  if value is None or isinstance(value, str | int | bool):
    return value
  if isinstance(value, float):
    if math.isfinite(value):
      return value
    return None
  if isinstance(value, Mapping):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
    return [_jsonable(item) for item in value]
  if is_dataclass(value):
    return _jsonable(value.__dict__)
  return str(value)
