from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import zipfile

import polars as pl
import pytest

from tabdat_synth.domain.contracts import ModelBackendSpec
from tabdat_synth import generate_from_spec, load_generation_spec


def _example_spec_path() -> Path:
  return Path(__file__).resolve().parents[2] / "docs/examples/de_synpuf_beneficiary_phase3.yaml"


def _dotenv_value(key: str) -> str | None:
  dotenv_path = Path(__file__).resolve().parents[2] / ".env"
  if not dotenv_path.exists():
    return None
  for line in dotenv_path.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
      continue
    name, value = stripped.split("=", 1)
    if name == key:
      return value
  return None


def test_generate_from_spec_runs_example_config_deterministically():
  spec = load_generation_spec(_example_spec_path())

  first = generate_from_spec(spec)
  second = generate_from_spec(spec)

  assert first.equals(second)
  assert first.height == spec.n_samples
  assert set(first.columns) == {
    "BENE_ESRD_IND",
    "BENE_SEX_IDENT_CD",
    "BENE_RACE_CD",
    "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
    "PLAN_CVRG_MOS_NUM",
    "SP_DIABETES",
    "SP_CHF",
    "BENE_SMI_CVRAGE_TOT_MONS",
  }

  assert set(first["BENE_SEX_IDENT_CD"].cast(pl.String).to_list()) <= {"1", "2"}
  assert set(first["BENE_RACE_CD"].cast(pl.String).to_list()) <= {"1", "2", "3", "5"}
  assert set(first["BENE_ESRD_IND"].cast(pl.String).to_list()) <= {"0", "Y"}
  assert set(first["SP_CHF"].cast(pl.String).to_list()) <= {"1", "2"}
  assert set(first["SP_DIABETES"].cast(pl.String).to_list()) <= {"1", "2"}
  assert first.select(pl.col("BENE_HMO_CVRAGE_TOT_MONS").is_null().any()).item() is False
  assert first.select(pl.col("BENE_HI_CVRAGE_TOT_MONS").is_between(0, 12).all()).item()
  assert first.select(pl.col("BENE_SMI_CVRAGE_TOT_MONS").is_between(0, 12).all()).item()
  assert first.select(pl.col("BENE_HMO_CVRAGE_TOT_MONS").is_between(0, 12).all()).item()
  assert first.select(pl.col("PLAN_CVRG_MOS_NUM").is_between(0, 12).all()).item()


@pytest.mark.local_data
def test_synpuf_local_zip_generation_smoke():
  if os.environ.get("RUN_LOCAL_DATA_TESTS") != "1":
    pytest.skip("Set RUN_LOCAL_DATA_TESTS=1 to run private/local DE-SynPUF smoke tests.")

  synpuf_dir = os.environ.get("SYNPUF_DIR") or _dotenv_value("SYNPUF_DIR")
  if not synpuf_dir:
    pytest.skip("SYNPUF_DIR is not configured.")

  zip_path = (
    Path(synpuf_dir)
    / "zip_files"
    / "de1_0_2010_beneficiary_summary_file_sample_18.zip"
  )
  if not zip_path.exists():
    pytest.skip(f"DE-SynPUF beneficiary zip not found: {zip_path}")

  with zipfile.ZipFile(zip_path) as archive:
    member_name = archive.namelist()[0]

  spec = load_generation_spec(_example_spec_path())
  local_spec = replace(
    spec,
    source_path=zip_path,
    source_member=member_name,
    n_samples=8,
    model_backend=ModelBackendSpec(
      kind="lightgbm",
      params={"n_estimators": 1, "verbosity": -1},
    ),
  )

  result = generate_from_spec(local_spec)

  assert result.height == 8
  assert result.select(pl.col("PLAN_CVRG_MOS_NUM").is_between(0, 12).all()).item()
  assert set(result["BENE_ESRD_IND"].cast(pl.String).to_list()) <= {"0", "Y"}
