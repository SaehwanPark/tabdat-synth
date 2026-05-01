from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
  src_path = PROJECT_ROOT / "src"
  if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

  from tabdat_synth import generate_from_spec, load_generation_spec

  spec_path = PROJECT_ROOT / "docs/examples/tiny_coefficient_outcomes.yaml"
  generated = generate_from_spec(load_generation_spec(spec_path))
  print(f"generated {generated.height} rows x {len(generated.columns)} columns")
  print(generated.head())


if __name__ == "__main__":
    main()
