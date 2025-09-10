<h1 align="center">clustools</h1>

<br/>

<!-- Project Badges -->
[![Documentation Status][docs-read-badge]][docs-read-url]
[![License - BSD-3-Clause][license-badge]][license-url]<br/>
[![PyPI - Version][pypi-badge]][pypi-url] [![PyPI - Downloads][pypi-downloads]][pypi-url] [![PyPI - Python Version][python-versions]][pypi-url]<br/>
[![CI - Tests][ci-badge]][ci-url] [![CD - Publish][cd-badge]][cd-url] [![GitHub stars][stars-badge]][github-url] [![GitHub forks][forks-badge]][github-url] [![GitHub issues][issues-badge]][issues-url]<br/>
[![Linting - Ruff][ruff-badge]][ruff-url]
[![Formatter - Ruff][formatter-ruff-badge]][ruff-url]
[![Types - Mypy][mypy-badge]][mypy-url]<br/>

<h3 align="center">
  <a href="https://github.com/psolsfer/clustools">GitHub</a>
  &middot;
  <a href="https://pypi.org/project/clustools/">PyPI</a>
  &middot;
  <a href="https://clustools.readthedocs.io/en/stable/">Docs</a>
  &middot;
  <a href="https://github.com/psolsfer/clustools/issues">Issues</a>
</h3>

<h3 align="center">
  A lightweight Python package providing essential clustering utilities for clustering workflows.
</h3>

<br/>

---

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
```
Install the prek git hook
```bash
uv run prek install
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

This package was created with [Copier](https://github.com/copier-org/copier) and the [Copier PyPackage uv](https://github.com/psolsfer/uvcopier) project template.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[docs-read-badge]: https://readthedocs.org/projects/clustools/badge/?version=stable&style=for-the-badge
[docs-read-url]: https://clustools.readthedocs.io/en/stable/

[license-badge]: https://img.shields.io/pypi/l/clustools.svg?style=for-the-badge
[license-url]: https://spdx.org/licenses/BSD-3-Clause.html

[pypi-badge]: https://img.shields.io/pypi/v/clustools.svg?logo=pypi&label=PyPI&logoColor=gold&style=for-the-badge
[pypi-url]: https://pypi.org/project/clustools/

[pypi-downloads]: https://img.shields.io/pypi/dm/clustools.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold&style=for-the-badge
[python-versions]: https://img.shields.io/pypi/pyversions/clustools.svg?logo=python&label=Python&logoColor=gold&style=for-the-badge

[ci-badge]: https://img.shields.io/github/actions/workflow/status/psolsfer/clustools/test-push-pr.yml?style=for-the-badge
[ci-url]: https://github.com/psolsfer/clustools/actions/workflows/test-push-pr.yml

[cd-badge]: https://img.shields.io/github/actions/workflow/status/psolsfer/clustools/python-publish.yml?style=for-the-badge
[cd-url]: https://github.com/psolsfer/clustools/actions/workflows/python-publish.yml

[stars-badge]: https://img.shields.io/github/stars/psolsfer/clustools.svg?style=for-the-badge
[forks-badge]: https://img.shields.io/github/forks/psolsfer/clustools.svg?style=for-the-badge

[issues-badge]: https://img.shields.io/github/issues/psolsfer/clustools.svg?style=for-the-badge
[issues-url]: https://github.com/psolsfer/clustools/issues

[github-url]: https://github.com/psolsfer/clustools

[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json&style=for-the-badge
[ruff-url]: https://github.com/astral-sh/ruff
[formatter-ruff-badge]: https://img.shields.io/badge/Ruff%20Formatter-checked-blue.svg?style=for-the-badge

[mypy-badge]: https://img.shields.io/badge/mypy%20-%20checked-blue?style=for-the-badge
[mypy-url]: https://mypy-lang.org/
