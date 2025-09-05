<style>
p[align="center"] {
  margin: 0px 0;  /* adjust as needed */
}
</style>

# clustools

<!-- Project Badges -->
<p align="center">
      <a href="https://clustools.readthedocs.io/en/stable/">
        <img src="https://readthedocs.org/projects/clustools/badge/?version=stable" alt="Documentation Status"/>
      </a>
    <a href="https://pypi.org/project/clustools/">
      <img src="https://img.shields.io/pypi/v/clustools.svg?logo=pypi&label=PyPI&logoColor=gold" alt="PyPI - Version"/>
    </a>
    <a href="https://pypi.org/project/clustools/">
      <img src="https://img.shields.io/pypi/dm/clustools.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold" alt="PyPI - Downloads"/>
    </a>
    <a href="https://pypi.org/project/clustools/">
      <img src="https://img.shields.io/pypi/pyversions/clustools.svg?logo=python&label=Python&logoColor=gold" alt="PyPI - Python Version"/>
    </a>
</p>

<p align="center">
  <a href="https://github.com/psolsfer/clustools/actions/workflows/test-push-pr.yml">
    <img src="https://github.com/psolsfer/clustools/actions/workflows/test-push-pr.yml/badge.svg" alt="CI - Test"/>
  </a>
  <a href="https://github.com/psolsfer/clustools/actions/workflows/python-publish.yml">
    <img src="https://github.com/psolsfer/clustools/actions/workflows/python-publish.yml/badge.svg" alt="CD - Build"/>
  </a>
  <a href="https://github.com/psolsfer/clustools">
    <img src="https://img.shields.io/github/stars/psolsfer/clustools.svg?style=social" alt="GitHub stars"/>
  </a>
  <a href="https://github.com/psolsfer/clustools">
    <img src="https://img.shields.io/github/forks/psolsfer/clustools.svg?style=social" alt="GitHub forks"/>
  </a>
  <a href="https://github.com/psolsfer/clustools/issues">
    <img src="https://img.shields.io/github/issues/psolsfer/clustools.svg" alt="GitHub issues"/>
  </a>
</p>

<p align="center">
  <a href="https://spdx.org/licenses/BSD-3-Clause.html">
    <img src="https://img.shields.io/pypi/l/clustools.svg" alt="License - BSD-3-Clause"/>
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" alt="Linting - Ruff"/>
  </a>
    <a href="https://github.com/astral-sh/ruff">
      <img src="https://img.shields.io/badge/Ruff%20Formatter-checked-blue.svg" alt="Ruff formatter"/>
    </a>
  <a href="https://mypy-lang.org/">
    <img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="Types - Mypy"/>
  </a>
</p>

<p align="center" style="margin: 20px 0;">
  <a href="https://github.com/psolsfer/clustools">GitHub</a>
  &middot;
  <a href="https://pypi.org/project/clustools/">PyPI</a>
  &middot;
  <a href="https://clustools.readthedocs.io/en/stable/">Docs</a>
  &middot;
  <a href="https://github.com/psolsfer/clustools/issues">Issues</a>
</p>

A lightweight Python package that extends scikit-learn's clustering ecosystem with additional algorithms and utilities. Features sklearn-compatible wrappers for Fuzzy C-Means, Faiss-based clustering, and supplementary functions for comprehensive clustering workflows.

<div style="margin: 20px 0;">
  <hr/>
</div>

## Features

TODO

## Installation

### From PyPI

```bash
pip install clustools
```

or

```bash
uv add clustools
```

### From Source

```bash
git clone https://github.com/psolsfer/clustools.git
cd clustools
uv sync
```

## Usage

### Python API

```python
import clustools

# TODO: Add usage examples
```

## Development

### Setup

```bash
git clone https://github.com/psolsfer/clustools.git
cd clustools
uv sync
uv run pre-commit install
```

### Running Tests
```bash
uv run pytest
```

### Code Quality

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src/clustools
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [Copier PyPackage uv](https://github.com/psolsfer/copier-pypackage-uv) project template.
