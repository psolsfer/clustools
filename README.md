# clustools

----
| | |
| --- | --- |
| **Docs** | [![Documentation Status](https://readthedocs.org/projects/clustools/badge/?version=stable)](https://clustools.readthedocs.io/en/stable/) |
| **Package** | [![PyPI - Version](https://img.shields.io/pypi/v/clustools.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.python.org/pypi/clustools) [![PyPI - Downloads](https://img.shields.io/pypi/dm/clustools.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.python.org/pypi/clustools) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/clustools.svg?logo=python&label=Python&logoColor=gold)](https://pypi.python.org/pypi/clustools) |
| **CI/CD** | [![CI - Test](https://github.com/psolsfer/clustools/actions/workflows/test-push-pr.yml/badge.svg)](https://github.com/psolsfer/clustools/actions/workflows/test-push-pr.yml) [![CD - Build](https://github.com/psolsfer/clustools/actions/workflows/python-publish.yml/badge.svg)](https://github.com/psolsfer/clustools/actions/workflows/python-publish.yml) |
| **GitHub** |  [![clustools](https://img.shields.io/badge/GitHub-clustools-blue.svg)](https://github.com/psolsfer/clustools) [![Forks](https://img.shields.io/github/forks/psolsfer/clustools.svg)](https://github.com/psolsfer/clustools) [![Stars](https://img.shields.io/github/stars/psolsfer/clustools.svg)](https://github.com/psolsfer/clustools) [![Issues](https://img.shields.io/github/issues/psolsfer/clustools.svg)](https://github.com/psolsfer/clustools) |
| **Code style** | [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![code style - Ruff formatter](https://img.shields.io/badge/Ruff%20Formatter-checked-blue.svg)](https://github.com/astral-sh/ruff) [![types - Mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/) |
| **License** | [![License - BSD-3-Clause-Clear](https://img.shields.io/pypi/l/clustools.svg)](https://spdx.org/licenses/BSD-3-Clause-Clear.html) |

A lightweight Python package that extends scikit-learn's clustering ecosystem with additional algorithms and utilities. Features sklearn-compatible wrappers for Fuzzy C-Means, Faiss-based clustering, and supplementary functions for comprehensive clustering workflows.

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
