# probLM-solver Codebase Assessment

## Project Overview
**probLM-solver** is a Python library built on top of `llama.cpp` that generates and performs statistical analysis on language model outputs.

- **Repository**: https://github.com/muse-writes/probLM-solver
- **Author**: Clio (d.k.johnson@lancaster.ac.uk)
- **Python Version**: >=3.13, <4.0
- **Build System**: uv_build
- **Status**: Version 1.3.0

---

## Directory Structure
```
probLM-solver/
├── src/problm_solver/          # Main source code (installed as `problm_solver`)
│   ├── __init__.py              # Package initialisation (empty docstring)
│   ├── adjust_probs.py          # GenerationContext, AdjustFn, adjustment functions, BranchSampler hierarchy
│   ├── cli.py                   # Command line interface
│   ├── constants.py             # Memory-size constants (KIB, MIB, GIB)
│   ├── llama_interface.py       # llama.cpp wrapper interface
│   ├── data.py                  # LLM output data containers and serialisation
│   ├── utils.py                 # Internal helpers (_as_rng, TqdmHandler)
│   └── analysis/                # Statistical analysis subpackage
│       ├── __init__.py          # Re-exports public analysis API
│       ├── tokenizer.py         # Tokenizer ABC and implementations
│       └── probabilities.py     # prob_of_token and sample_from_logprobs utilities
├── tests/
│   ├── __init__.py
│   ├── test_import.py           # Basic import test
│   ├── test_data.py             # Tests for data.py
│   ├── test_cli.py              # Tests for cli.py
│   ├── test_llama_interface.py  # Tests for llama_interface.py
│   ├── test_probabilities.py    # Tests for analysis/probabilities.py
│   └── test_adjust_probs.py    # Tests for adjust_probs.py
├── docs/
│   ├── index.md                 # MkDocs landing page (legacy, kept in place)
│   ├── conf.py                  # Sphinx configuration
│   ├── index.rst                # Sphinx landing page with toctree
│   ├── api.rst                  # Sphinx API reference (one automodule section per module)
│   ├── tutorials.rst            # User-facing scripting tutorials
│   └── Makefile                 # `make html` builds; `make clean` wipes _build/
├── pyproject.toml               # Project configuration & dependencies
├── Dockerfile                   # Multi-stage Docker build (dev + app targets)
├── docker-compose.yml           # Compose configuration
├── mkdocs.yml                   # MkDocs config (superseded by Sphinx; retained)
├── README.md                    # Project README
└── .gitignore                   # Standard Python gitignore
```

---

## Architecture

### Dependency Direction
```
cli.py → ModelInstance (llama_interface.py) → LLMOutputData / LLMTokenData / LLMNextTokenData / LLMOutputDataFull / Hyperparams (data.py)
                                             → LlamaTokenizer (analysis/tokenizer.py)
                                             → prob_of_token, sample_from_logprobs (analysis/probabilities.py)
                                             → AdjustFn, GenerationContext (adjust_probs.py)
cli.py → SampleLowTemp, SamplePowerDist, MetropolisSampler (adjust_probs.py)
cli.py → LLMOutputData (data.py)
adjust_probs.py → _as_rng (utils.py)
adjust_probs.py → prob_of_token (analysis/probabilities.py)
analysis/probabilities.py → _as_rng (utils.py)
analysis/tokenizer.py → (no intra-package deps)
data.py → (no intra-package deps)
utils.py → tqdm (external)
constants.py → (no intra-package deps)
cli.py → TqdmHandler (utils.py)
```

### Data Flow
1. User selects a GGUF model file from `~/.problm-solver/models/`
2. User enters a prompt, then selects a function (1–4)
3. `ModelInstance` is created; `logits_all=True` for functions 2, 3, and 4
4. **Text responses** (1): `model.generate_data(n)` → `LLMOutputData`; saved to `responses/`
5. **Token + probability responses** (2): `model.query_log_probs()` → `LLMTokenData`; saved to `probabilities/`
6. **Low-temperature generation** (3): `model.generate_adjusted(top_k, top_p, SampleLowTemp(alpha), max_tokens, alpha=alpha)` → `LLMOutputDataFull`; saved to `responses/`
7. **Power MCMC generation** (4): `model.generate_adjusted(top_k, top_p, SamplePowerDist(alpha, depth, MetropolisSampler()), max_tokens, alpha=alpha, sampling_method=..., branch_sampler=...)` → `LLMOutputDataFull`; saved to `responses/`

---

## Dependencies

### Runtime (`dependencies` in pyproject.toml)
- `llama-cpp-python`: LLM inference via llama.cpp
- `numpy`: Array operations
- `tqdm >= 4.67.3`: Progress bars; also used by `TqdmHandler` in `utils.py`
- `poethepoet`: Task runner (kept runtime so it is available inside the Docker app container)

### Development (`dependency-groups.dev`)
- `pytest >= 8.3.4` + `pytest-mock`, `pytest-xdist`: Testing
- `coverage[toml] >= 7.6.10`: Code coverage (minimum 50%)
- `ruff >= 0.9.2`: Linting & formatting
- `pre-commit >= 4.0.1`: Git hooks
- `commitizen >= 4.3.0`: Semantic versioning
- `sphinx >= 9.1.0` + `sphinx-autodoc-typehints`, `shibuya`: Reference documentation
- `mkdocs-material >= 9.5.21`: MkDocs (superseded by Sphinx; retained)
- `typeguard >= 4.4.1`, `ty >= 0.0.6`: Type checking
- `ipython`, `ipykernel`, `ipywidgets`: Jupyter support
- `codespell >= 2.4.1`: Spell checking

---

## Code Quality & Configuration

### Testing
- **Framework**: pytest with doctest modules
- **Coverage**: Minimum 50% (`fail_under = 50`)
- **Options**: Exit on first failure, verbose (v=2), JUnit XML reports
- **Paths**: `src/`, `tests/`
- **Test files**: `test_import.py`, `test_data.py`, `test_cli.py`, `test_llama_interface.py`, `test_probabilities.py`, `test_adjust_probs.py`
- **Total tests**: 225
- **Doctests**: `WordTokenizer` and `LlamaTokenizer` have inline doctests collected by `--doctest-modules`
- **Mocking**:
  - `llama_cpp.Llama` is patched at construction time in all `ModelInstance` tests; `_make_llama_mock` provides metadata and `n_ctx` so `__init__`'s cache-sizing logic runs correctly
  - `llama_cpp.LlamaRAMCache` is patched in `test_cache_sized_from_model_metadata`
  - `Jinja2ChatFormatter` is patched at `problm_solver.llama_interface.Jinja2ChatFormatter` in `TestFormatChatPrompt`
  - Low-level `eval`/`scores`/`save_state`/`load_state` methods are mocked with `side_effect` functions that maintain `n_tokens` state, mirroring real llama_cpp behaviour; `scores` is a real numpy array with known values at the relevant `n_tokens - 1` row
  - Tests that check **exact log-prob or probability values** and require deterministic token selection must patch `problm_solver.llama_interface._as_rng` to return a `MagicMock` whose `.gumbel` returns `np.zeros(vocab_size)`. **Do not** use `patch('numpy.random.gumbel', ...)` — the code uses `numpy.random.Generator.gumbel` (new-style API), which is unaffected by that patch. Tests that only check structure, counts, or types do not need gumbel control.
  - `_format_chat_prompt`, `sample_from_logprobs`, and `prob_of_token` are all patched in the `gen_adj_model` fixture via `contextlib.ExitStack`
  - `adjust_fn` receives a `GenerationContext`; `MagicMock.call_args_list` inspection reads `.prev_probs` from the context object
  - `GenerationContext` fixtures in `test_adjust_probs.py` supply a `query_branch=MagicMock(return_value=-1.5)` field
  - `mock_sampler` in `TestSamplePowerDistCall` sets `future_logprob.return_value = 0.0` so output values are Python `float` rather than `MagicMock`
  - CLI filesystem interactions use `monkeypatch` against `tmp_path`

### Linting (Ruff)
- **Line Length**: 100 characters
- **Target Python**: 3.13
- **Docstring Convention**: reStructuredText / PEP 257 (`:param:`, `:returns:` directives)
- **Quotes**: Single (inline and multiline)
- **Imports**: Absolute only (`ban-relative-imports = "all"`)

### Versioning
- Conventional Commits enforced via Commitizen
- Current version: **1.3.0**

---

## Non-Obvious Invariants and Pitfalls

These are recurring sources of bugs in this codebase that an agent should be aware of before making changes.

1. **`self._llm.n_tokens` vs the `top_k` parameter** (`llama_interface.py`): `self._llm.n_tokens` is the llama.cpp model's current token count (changes with every `eval()` call). The parameter formerly named `n_tokens` in `generate_adjusted` has been renamed to `top_k`. Do not confuse them — `scores[self._llm.n_tokens - 1]` is always the correct logit row.

2. **`special=True`/`special=False` in `generate_adjusted`** (`llama_interface.py`): Two separate `tokenize()` calls serve different purposes. The `token_ids` check uses `special=True` so that `<|im_end|>` is recognised as `eos_id` and generation terminates. The `tokenize_token` lambda passed to `GenerationContext` uses `special=False` so that special token strings are not injected as actual special token IDs into branch contexts (which corrupts branch evaluation).

3. **Save/restore around `adjust_fn`** (`llama_interface.py`): `SamplePowerDist` calls `query_branch` internally, which calls `reset()` + `eval()` on the shared `_llm`. This corrupts the KV-cache state. `generate_adjusted` saves state immediately before `adjust_fn(ctx)` and restores it immediately after, so the subsequent incremental `eval(token_ids)` always appends to the correct generation context.

4. **`past_lp` is a constant shift** (`adjust_probs.py`): In `SamplePowerDist.__call__`, `past_lp = alpha * sum(log(prev_probs))` is computed once and added to every candidate's score. Because `sample_from_logprobs` and `prob_of_token` both apply shift-invariant softmax (`lp -= lp.max()`), `past_lp` has no effect on which token is selected or on the stored per-token probabilities.

5. **`future_logprob` discards burn-in samples** (`adjust_probs.py`): `MetropolisSampler.future_logprob` uses only `branch_log_probs[equil_branches:]`. With default settings the minimum number of post-equilibration samples is 2, giving a high-variance estimate. This is intentional (burn-in removal) but means the branch estimate quality is sensitive to `equil_branches` and `max_branches`.

6. **`numpy.random.gumbel` patch does not reach `Generator.gumbel`**: The codebase uses `np.random.default_rng(seed).gumbel(...)` (new-style Generator API). Patching `numpy.random.gumbel` targets the legacy API and has no effect. Tests that need deterministic sampling must patch `problm_solver.llama_interface._as_rng`.

---

## Known Issues — Active

1. **`ui_save_data()` discard logic unreachable** (`cli.py`): `input()` always returns a `str`; `if fname is not None` is always `True`, so the `print('Aborted saving.')` branch can never execute. Should be `if fname` (truthy check for empty string).

2. **Unreachable guard in `ui_gen_data()`** (`cli.py`): `data_size = int(input(...))` either succeeds (always an `int`) or raises; the `while not isinstance(data_size, int)` loop below it can never be reached.

3. **Hard-coded model parameters** (`llama_interface.py`): `n_ctx=2048` and `max_tokens=512` are not user-configurable.

4. **`FBT001` on `_LlamaInterface` protocol** (`analysis/tokenizer.py`): `add_bos: bool` and `special: bool` are flagged as boolean positional arguments. These mirror the external `llama_cpp.Llama` API and cannot be made keyword-only.

5. **Equilibration period not user-configurable** (`adjust_probs.py`): `should_continue()` discards `branch_log_probs[:equil_branches]` as a fixed equilibration period. Whether this is the right threshold, and whether users should be able to control it separately from `equil_branches`, is an open design question (TODO in source).

6. **`top_p` stubbed** (`llama_interface.py`): resolved — see below.

7. **`ui_generate_low_temp` missing `sampling_method`** (`cli.py`): does not pass `sampling_method` to `generate_adjusted()`; the label stored in `LLMOutputDataFull` is auto-derived as `'SampleLowTemp'` rather than a clean human-readable string.

8. **`get_adjusted_path()` extension** (`cli.py`): returns a `.jsonl` path, but `LLMOutputDataFull.write()` produces single-record JSON; should be `.json`.

9. **`_written` comment indentation** (`data.py`): the inline comment `# Unsaved data state tracking variable.` before `_written` is at column 0 inside the `LLMOutputDataFull` class body; should be indented 4 spaces.

10. **`query_next` lambda resets model state** (`llama_interface.py`): the `query_next` callable in `GenerationContext` is bound to `query_log_probs_next_token`, which calls `reset()` + `eval()` on the shared `_llm` instance. If an `adjust_fn` calls `query_next` during `generate_adjusted`, it will corrupt the incremental eval state that the generation loop depends on. The same issue applied to `query_branch` and is now resolved (see below), but `query_next` remains unfixed as `SamplePowerDist` does not currently use it.
