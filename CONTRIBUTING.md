# Contributing to probLM-solver

If you're reading this, you may be considering contributing to probLM-solver. Thank you for making it this far, your interest is very welcome!

## Code of Conduct

Before developing for probLM-solver, please see our [Code of Conduct](./CODE_OF_CONDUCT.md)

## Bug Reports

Please raise an issue describing the bug if you encounter one. probLM-solver is set up to use Python's `logging` module, so if possible, include it in the script you're using with the verbosity set to `logging.DEBUG`. Share the output in your issue.

## Pull Request Guidelines

Pull requests (PRs) must be clear in scope, and ideally linked to a previously opened issue.

AI usage is welcome, but all PRs must be human readable (in a reasonable time-frame) for maintainability.

A PR should contain no more than one or two features in any number of commits.

All commit messages should adhere to the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard.

## Development Setup

### Prerequisites

- Python 3.13 or above
- pip
- uv

### Getting Started

First, fork probLM-solver, and then clone it:
```Bash
git clone git@github.com:[your_username]/probLM-solver.git
```

Set this repository as upstream:
```Bash
git remote add upstream https://github.com/muse-writes/probLM-solver
```

Create a branch for your changes,
```Bash
git checkout -b [your-feature-name]
```
Branch names should be short and descriptive.

Use `uv` from the project's directory to create and synchronize the virtual environment,
```Bash
uv sync --python 3.13 --all-extras
```
and then source it,
```Bash
source .venv/bin/activate
```

This virtual environment can be used in a shebang for Python scripts that work from anywhere else:
```Python
#!/path/to/probLM-solver/.venv/bin/python

from problm_solver.llama_interface import Model
...
```

If you are aiming to run your script on a GPU, you may need to reinstall `llama-cpp-python` with the correct CUDA environment flags as seen in [the README](./README.md#usage). The minimum `llama-cpp-python` version requirement is `0.3.23`.

### Continuous Integration

Whilst in the development virtual environment, run the unit test suite using,
```Bash
poe test
```

It's a good idea to run this prior to making any changes to verify everything passes and generate an initial CI file.

### Before Submitting a PR

Make sure you have run the test suite using `poe test`.

If you have updated any documentation in your PR, please remake the docs using `poe docs`. You can verify these changes by exploring the docs locally using `poe docs --serve`.

If any unit tests need changing to accommodate your changes, make those changes and draw attention to them in the PR itself.
