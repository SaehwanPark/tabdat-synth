from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def _dependency_name(requirement: str) -> str:
  match = re.match(r"[A-Za-z0-9_.-]+", requirement)
  if match is None:
    raise AssertionError(f"Could not parse dependency requirement: {requirement}")
  return match.group(0).lower().replace("_", "-")


def _top_level_imported_roots(path: Path) -> set[str]:
  tree = ast.parse(path.read_text())
  roots: set[str] = set()
  for node in tree.body:
    if isinstance(node, ast.Import):
      roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      roots.add(node.module.split(".", maxsplit=1)[0])
  return roots


def test_public_runtime_dependencies_exclude_dev_only_packages():
  pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text())
  runtime_dependencies = {
    _dependency_name(requirement)
    for requirement in pyproject["project"]["dependencies"]
  }

  assert runtime_dependencies.isdisjoint({"pytest", "python-dotenv", "tqdm"})
  assert {"lightgbm", "numpy", "pandas", "polars", "pyarrow"}.issubset(
    runtime_dependencies
  )
  assert pyproject["project"]["optional-dependencies"]["env"] == [
    "python-dotenv>=1.2.2"
  ]


def test_dev_dependency_group_contains_test_lint_and_build_check_tools():
  pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text())
  dev_dependencies = {
    _dependency_name(requirement)
    for requirement in pyproject["dependency-groups"]["dev"]
  }

  assert {"pytest", "ruff", "twine"}.issubset(dev_dependencies)


def test_license_metadata_and_file_are_present():
  root = _repo_root()
  pyproject = tomllib.loads((root / "pyproject.toml").read_text())

  assert pyproject["project"]["license"] == "Apache-2.0"
  assert pyproject["project"]["license-files"] == ["LICENSE"]
  assert "Apache License" in (root / "LICENSE").read_text()


def test_python_support_metadata_targets_python_312_and_313():
  pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text())
  classifiers = set(pyproject["project"]["classifiers"])

  assert pyproject["project"]["requires-python"] == ">=3.12"
  assert "Programming Language :: Python :: 3.12" in classifiers
  assert "Programming Language :: Python :: 3.13" in classifiers
  assert pyproject["tool"]["mypy"]["python_version"] == "3.12"


def test_project_version_matches_package_version():
  root = _repo_root()
  pyproject = tomllib.loads((root / "pyproject.toml").read_text())
  version_module = ast.parse((root / "src/tabdat_synth/version.py").read_text())
  assignments = [
    node
    for node in version_module.body
    if isinstance(node, ast.Assign)
    and any(
      isinstance(target, ast.Name) and target.id == "__version__"
      for target in node.targets
    )
  ]

  assert len(assignments) == 1
  assert isinstance(assignments[0].value, ast.Constant)
  assert pyproject["project"]["version"] == assignments[0].value.value


def test_dotenv_is_optional_at_the_io_boundary():
  imported_roots = _top_level_imported_roots(
    _repo_root() / "src/tabdat_synth/interfaces/io.py"
  )

  assert "dotenv" not in imported_roots


def test_docs_include_pypi_uv_install_commands():
  root = _repo_root()
  docs_text = "\n".join(
    [
      (root / "README.md").read_text(),
      (root / "docs/getting-started.md").read_text(),
      (root / "docs/user-manual.md").read_text(),
    ]
  )

  assert "uv add tabdat-synth" in docs_text
  assert "uv pip install tabdat-synth" in docs_text
  assert 'uv add "tabdat-synth[env]"' in docs_text
  assert "git+https://github.com/SaehwanPark/tabdat-synth.git" not in docs_text
  assert "Python 3.12" in docs_text
