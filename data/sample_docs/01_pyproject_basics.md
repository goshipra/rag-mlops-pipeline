# pyproject.toml Basics

`pyproject.toml` is the single, standardized configuration file for a modern
Python project. It was introduced by PEP 518 and later extended by PEP 621 to
cover project metadata, replacing the old trio of `setup.py`, `setup.cfg`,
and `MANIFEST.in` for most use cases.

A minimal `pyproject.toml` has three parts:

1. `[build-system]` — declares which tool builds your package (for example
   `setuptools`, `hatchling`, or `poetry-core`) and the minimum versions
   required to build it. This is what lets `pip install .` work without the
   installer needing to guess how your project is built.
2. `[project]` — the PEP 621 metadata table: `name`, `version`, `description`,
   `readme`, `requires-python`, `dependencies`, and `authors`. This is the
   section tools like `pip`, `build`, and `twine` read to know what your
   package is called and what it needs.
3. Tool-specific tables such as `[tool.pytest.ini_options]`,
   `[tool.ruff]`, or `[tool.mypy]` — every tool that wants project-level
   configuration gets its own namespaced table under `[tool.*]`, so a single
   file can configure the whole toolchain.

Example:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rag-mlops-pipeline"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["requests>=2.31"]
```

The key benefit of `pyproject.toml` over `setup.py` is that it is
declarative and static: tools can read your project's metadata and build
requirements without executing arbitrary Python code, which makes builds
faster, safer, and reproducible across machines.
