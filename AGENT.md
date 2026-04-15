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
├── src/problm-solver/          # Main source code
│   ├── __init__.py              # Package initialization (currently empty docstring)
│   ├── cli.py                   # Command line interface (partially incomplete)
│   ├── llama_interface.py        # llama.cpp wrapper interface
│   └── gen_data.py              # Data generation from LLM outputs
├── tests/
│   ├── __init__.py
│   └── test_import.py           # Basic import test (has bug - see issues)
├── docs/
│   └── index.md                 # Documentation (minimal)
├── pyproject.toml              # Project configuration & dependencies
├── Dockerfile                   # Docker setup
├── docker-compose.yml           # Compose configuration
├── mkdocs.yml                  # Documentation build config
├── README.md                   # Project README
└── .gitignore                  # Standard Python gitignore
```

---

## Core Modules

### 1. **llama_interface.py**
**Purpose**: Wrapper around llama.cpp for local model inference

**Key Classes**:
- `ModelInstance`: Represents a loaded GGUF model
  - `__init__(fname: str, context: str)`: Load model with context
  - `query() -> str`: Single query to model, returns string
  - `query_n_times(n: int) -> np.ndarray[str]`: Run same query N times
  - Hard-coded context window: `n_ctx=2048`
  - Hard-coded max tokens: `max_tokens=256`
  - Uses chat completion format with role='user'

**Dependencies**:
- `llama_cpp`: Llama class for model loading
- `numpy` / `numpy.typing`: Array handling

### 2. **gen_data.py**
**Purpose**: Data generation and management from LLM outputs

**Key Classes**:
- `LLMOutputData`: Generate, store, and read LLM outputs
  - `__init__(model: ModelInstance)`: Accepts a ModelInstance
  - `generate(n_samples: int)`: Query model N times (stores in `self.data`)
  - `write(fname=None)`: Save data to file in JSONL format (default: `[model]_[timestamp].jsonl`)
    - One JSON object per line with `id`, `prompt`, and `response` fields
    - Properly handles numpy arrays by iterating through responses
  - `self.written` flag: Tracks if data has been written
  - Guard: Prevents overwriting data without writing first

**Current Status**: Functional - `write()` method implemented with JSONL output

### 3. **cli.py**
**Purpose**: Command-line interface for user interaction

**Current Features**:
- `MODELS_DIR`: `~/.problm-solver/models/`
- `DATA_DIR`: `~/.problm-solver/datasets/`
- `ensure_models_dir()`: Creates models directory
- `ensure_data_dir()`: Creates datasets directory
- `list_models()`: Lists all `.gguf` files (sorted)
- `select_model()`: Interactive model selection UI
- `main()`: Main loop that:
  1. Ensures models directory exists
  2. Prompts user to select a model
  3. Gets user prompt
  4. Creates ModelInstance
  5. Prompts for number of samples
  6. Generates data via `LLMOutputData.generate()`
  7. Saves data to JSONL file with timestamped filename
  8. Uses timestamp format: `YYYY-MM-DD-HH:MM:SS`

**Current Status**: Fully functional workflow implemented

---

## Dependencies

### Runtime
- `poethepoet >= 0.32.1`: Task runner

### Development
- `pytest >= 8.3.4`: Testing
- `pytest-mock >= 3.14.0`: Mocking utilities
- `pytest-xdist >= 3.6.1`: Parallel testing
- `coverage >= 7.6.10`: Code coverage
- `ruff >= 0.9.2`: Linting & formatting
- `mkdocs-material >= 9.5.21`: Documentation
- `pre-commit >= 4.0.1`: Git hooks
- `commitizen >= 4.3.0`: Semantic versioning
- `typeguard >= 4.4.1`: Type checking
- `ipython`, `ipykernel`, `ipywidgets`, `ipynb`: Jupyter support
- `codespell >= 2.4.1`: Spell checking

### Implicit (not in pyproject.toml but used)
- `llama_cpp`: LLM inference engine (NEEDS TO BE ADDED)
- `numpy`: Array operations (NEEDS TO BE ADDED)

---

## Code Quality & Configuration

### Testing
- **Framework**: pytest with doctest modules
- **Coverage**: Minimum 50% required (`fail_under = 50`)
- **Options**: Exit on first failure, verbose output (v=2), JUnit XML reports
- **Test Locations**: `src/`, `tests/`

### Linting (Ruff)
- **Line Length**: 100 characters
- **Target Python**: 3.13
- **Convention**: NumPy docstrings
- **Formatting**: docstring-code-format enabled
- **Ignores**: Many (CPY, FIX, COM812, D203, D213, S101, etc.)

### Versioning & Commits
- **Standard**: Conventional Commits (enforced by Commitizen)
- **Semantic Versioning**: Auto-bumped via `cz bump`
- **Changelog**: Auto-generated (Keep A Changelog format)
- **Version Source**: UV-managed

### Documentation
- **Tool**: MkDocs with Material theme
- **Serves via**: `poe docs --serve`

---

## Known Issues & Incomplete Features

### High Priority
1. **Missing Dependencies**: `llama_cpp` and `numpy` not declared in `pyproject.toml`
   - Likely runtime failures when importing

2. **Test File Bug** (`tests/test_import.py`):
   - Line 4: `import problm-solver` is invalid Python syntax
   - Should be: `import problm_solver` (underscores, not hyphens)
   - Current test will fail to run

### Medium Priority
1. **Hard-coded Model Parameters**:
   - Context window (2048) and max tokens (256) hard-coded in `ModelInstance`
   - Should be configurable via constructor or config file

2. **No Error Handling**:
   - Model loading could fail silently
   - Network/file I/O errors not caught

3. **Documentation Minimal**:
   - `docs/index.md` is just a copy of README
   - No API documentation or usage examples

### Completed ✓
- ✓ `gen_data.py` `write()` method now fully implemented with JSONL output format
- ✓ CLI workflow now complete with data generation and saving
- ✓ Timestamp functionality added to cli.py

---

## Development Workflow

### Available Poe Tasks
```bash
poe lint           # Run pre-commit hooks on all files
poe test           # Run tests + coverage reporting
poe docs            # Build documentation
poe docs --serve    # Serve documentation locally (live reload)
poe serve           # Serve the app (currently placeholder)
```

### Git Workflow
- Conventional Commits enforced
- Bump version: `cz bump` (auto-updates changelog, creates git tag)
- Push: `git push origin main --tags`

### Development Environments Supported
1. **GitHub Codespaces** (click link in README)
2. **VS Code Dev Container** (with container volume)
3. **uv** (local Python 3.13 venv)
4. **PyCharm Dev Container**

---

## Next Steps / Recommendations for Development

### Critical (Must Do)
- [ ] Add `llama_cpp` and `numpy` to `pyproject.toml` dependencies
- [ ] Fix test import statement (use underscores)

### Important (Should Do)
- [ ] Add more comprehensive tests
- [ ] Make model parameters configurable (context window, max tokens)
- [ ] Add error handling for model loading/querying
- [ ] Expand documentation with API examples
- [ ] Add statistical analysis functions (currently only data generation)

### Nice to Have
- [ ] Add configuration file support (YAML/TOML for model settings)
- [ ] Add model caching/management utilities
- [ ] Add output format options (JSON, CSV, etc.)
- [ ] Performance profiling/optimization
- [ ] Support for batch processing across multiple models

---

## Last Git History (5 commits)
1. `db6492e` - feat: Added standard Python .gitignore, implemented simple CLI tool
2. `faf50e7` - feat: Added source files cli.py and llama_interface.py
3. `3345d7a` - build(template): grabbed a copier template
4. `a9e190c` - Init & README.md

---

## Architecture Notes

### Current Design
- **Separation of Concerns**: Good - LLM interface, data gen, and CLI are separate
- **Data Flow**: CLI → ModelInstance (llama_interface) → LLMOutputData (gen_data)
- **Model Loading**: Done once per CLI session via GGUF files from `~/.problm-solver/models/`

### Future Considerations
- Would likely need database for storing generated datasets
- Statistical analysis module not yet implemented (mentioned in project description)
- Scalability: Currently single-model, single-query mode
