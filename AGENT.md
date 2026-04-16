# probLM-solver Codebase Assessment

## Project Overview
**probLM-solver** is a Python library built on top of `llama.cpp` that generates and performs statistical analysis on language model outputs.

- **Repository**: https://github.com/muse-writes/probLM-solver
- **Author**: Clio (d.k.johnson@lancaster.ac.uk)
- **Python Version**: >=3.13, <4.0
- **Build System**: uv_build
- **Status**: Version 0.0.0 (early development)

---

## Directory Structure
```
probLM-solver/
├── src/problm_solver/          # Main source code (installed as `problm_solver`)
│   ├── __init__.py              # Package initialisation (empty docstring)
│   ├── cli.py                   # Command line interface
│   ├── llama_interface.py       # llama.cpp wrapper interface
│   └── data.py                  # LLM output data container and serialisation
├── tests/
│   ├── __init__.py
│   └── test_import.py           # Basic import test (has bug - see issues)
├── docs/
│   └── index.md                 # Documentation (minimal)
├── pyproject.toml               # Project configuration & dependencies
├── Dockerfile                   # Multi-stage Docker build (dev + app targets)
├── docker-compose.yml           # Compose configuration
├── mkdocs.yml                   # Documentation build config
├── README.md                    # Project README
└── .gitignore                   # Standard Python gitignore
```

---

## Core Modules

### 1. **data.py**
**Purpose**: Standalone data container for LLM output; handles serialisation and deserialisation.

**Key Classes**:
- `LLMOutputData`: Stores a prompt and its associated LLM responses
  - `__init__(prompt: str, data: npt.NDArray[Any])`: Constructed with raw data, no dependency on `ModelInstance`
  - `write(fname: str)`: Saves to JSONL format; each line has `id`, `prompt`, `response` fields; sets `self.written = True`
  - `read(fname: str)`: Loads from a JSONL file; populates `self.prompt`, `self.data`, and sets `self.written = True`
  - `self.written`: Tracks whether in-memory data is in sync with disk

**Dependencies**: `json`, `numpy`

---

### 2. **llama_interface.py**
**Purpose**: Wrapper around `llama_cpp` for local model inference. Owns the responsibility of generating `LLMOutputData`.

**Key Classes**:
- `ModelInstance`: Represents a loaded GGUF model
  - `__init__(fname: str, context: str)`: Loads model via `Llama`; hard-coded `n_ctx=2048`
  - `query() -> str`: Single inference call; hard-coded `max_tokens=512`; uses chat completion with `role='user'`
  - `query_n_times(n: int) -> npt.NDArray[Any]`: Calls `query()` N times, returns numpy array
  - `generate_data(n_samples: int) -> LLMOutputData`: Calls `query_n_times()` and wraps result in `LLMOutputData`

**Dependencies**: `llama_cpp`, `numpy`, `problm_solver.data`

---

### 3. **cli.py**
**Purpose**: Interactive command-line interface for the full generate-and-save workflow.

**Constants**:
- `PROBLM_DIR`: `~/.problm-solver/`
- `MODELS_DIR`: `~/.problm-solver/models/`
- `DATA_DIR`: `~/.problm-solver/datasets/`

**Functions**:
- `ensure_models_dir() / ensure_data_dir()`: Create directories if absent
- `list_models() -> list[Path]`: Sorted list of `.gguf` files in `MODELS_DIR`
- `get_data_path(model_path: Path) -> Path`: Builds a timestamped JSONL output path using UTC time
- `ui_select_model() -> Path`: Interactive model picker
- `ui_save_data(fname: str, data: LLMOutputData)`: Prompts user to confirm saving; calls `data.write()`
- `main()`: Full workflow — select model → enter prompt → generate data → save to JSONL

**Dependencies**: `problm_solver.llama_interface`, `problm_solver.data`

---

## Architecture

### Dependency Direction
```
cli.py → ModelInstance (llama_interface.py) → LLMOutputData (data.py)
cli.py → LLMOutputData (data.py)
```

`LLMOutputData` has no dependency on `ModelInstance`. `ModelInstance` constructs and returns `LLMOutputData` via `generate_data()`. This is an intentional inversion from an earlier design where `LLMOutputData` depended on `ModelInstance`.

### Data Flow
1. User selects a GGUF model file from `~/.problm-solver/models/`
2. `ModelInstance` is created with the model path and user-supplied prompt
3. `model.generate_data(n)` queries the LLM N times and returns an `LLMOutputData`
4. `LLMOutputData.write()` serialises results to a timestamped JSONL file in `~/.problm-solver/datasets/`

---

## Dependencies

### Runtime (`dependencies` in pyproject.toml)
- `llama-cpp-python`: LLM inference via llama.cpp
- `numpy`: Array operations
- `poethepoet`: Task runner — kept as a runtime dependency so it is available inside the Docker app container (installed with `--no-dev`)

### Development (`dependency-groups.dev`)
- `pytest >= 8.3.4` + `pytest-mock`, `pytest-xdist`: Testing
- `coverage[toml] >= 7.6.10`: Code coverage (minimum 50%)
- `ruff >= 0.9.2`: Linting & formatting
- `pre-commit >= 4.0.1`: Git hooks
- `commitizen >= 4.3.0`: Semantic versioning
- `mkdocs-material >= 9.5.21`: Documentation
- `typeguard >= 4.4.1`, `ty >= 0.0.6`: Type checking
- `ipython`, `ipykernel`, `ipywidgets`: Jupyter support
- `codespell >= 2.4.1`: Spell checking

---

## Docker Setup

### Dockerfile (multi-stage)
- **`dev` stage**: Based on `ghcr.io/astral-sh/uv:python3.13-trixie`; sets up venv at `/opt/venv`; intended for VS Code Dev Containers
- **`app` stage**: Based on `python:3.13-slim`; runs `uv sync --no-dev --no-editable --frozen`; venv at `/workspaces/problm-solver/.venv`; ENTRYPOINT is `.venv/bin/poe`, CMD is `serve`

### docker-compose.yml
- **`devcontainer`**: Runs by default (`docker compose up`); mounts `..:/workspaces`
- **`app`**: Requires explicit profile (`docker compose up app`); has `stdin_open: true` and `tty: true` for interactive CLI; mounts `~/.problm-solver:/home/user/.problm-solver` for model and dataset persistence

### Running the App
```sh
# From host
docker compose up app

# Equivalent from inside Dev Container
poe serve
```

---

## Poe Tasks (`[tool.poe.tasks]`)

The poe executor is set to `type = "virtualenv"` so tasks resolve executables from the project venv rather than the system PATH.

```bash
poe serve           # Run the CLI (calls the `problm-solver` entry point)
poe lint            # Run pre-commit hooks on all files
poe test            # Run tests + coverage report + coverage XML
poe docs            # Build documentation
poe docs --serve    # Serve documentation locally with live reload
```

### Entry Point
`[project.scripts]` declares:
```
problm-solver = "problm_solver.cli:main"
```
This is installed into the venv's `bin/` by `uv sync` and is the target of `poe serve`.

---

## Code Quality & Configuration

### Testing
- **Framework**: pytest with doctest modules
- **Coverage**: Minimum 50% (`fail_under = 50`)
- **Options**: Exit on first failure, verbose (v=2), JUnit XML reports
- **Paths**: `src/`, `tests/`

### Linting (Ruff)
- **Line Length**: 100 characters
- **Target Python**: 3.13
- **Docstring Convention**: NumPy
- **Quotes**: Single (inline and multiline)
- **Imports**: Absolute only (`ban-relative-imports = "all"`)

### Versioning
- Conventional Commits enforced via Commitizen
- `cz bump` → updates `CHANGELOG.md`, bumps version, creates git tag

---

## Known Issues

### Active
1. **Test file bug** (`tests/test_import.py`):
   - `import problm-solver` is invalid Python syntax (hyphens not allowed in module names)
   - Should be `import problm_solver`

2. **Unreachable input validation loop** (`cli.py`, `main()`):
   - `data_size = int(input(...))` either succeeds (always an `int`) or raises an exception
   - The `while not isinstance(data_size, int)` guard below it can never be reached

3. **Hard-coded model parameters** (`llama_interface.py`):
   - `n_ctx=2048` and `max_tokens=512` are not configurable

### Resolved
- ✓ `LLMOutputData.read()` implemented
- ✓ `LLMOutputData` dependency on `ModelInstance` removed; `generate_data()` added to `ModelInstance`
- ✓ `llama-cpp-python` and `numpy` added to runtime dependencies
- ✓ `poethepoet` moved to runtime dependencies for Docker app container compatibility
- ✓ Poe executor changed from `simple` to `virtualenv`
- ✓ `poe serve` wired up to `problm-solver` entry point (was a placeholder echo)
- ✓ Intra-package imports converted from bare names to absolute (`problm_solver.*`)
- ✓ Docker `app` service configured with `stdin_open`, `tty`, and volume mount for model/data persistence
