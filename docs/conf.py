# Sphinx configuration for probLM-solver
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

# Make the src layout visible to autodoc without requiring an editable install.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

with (project_root / 'pyproject.toml').open('rb') as pyproject_file:
    pyproject = tomllib.load(pyproject_file)

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = pyproject['project']['name']
author = 'Clio Johnson'
release = pyproject['project']['version']
copyright = f'2024–{datetime.now(UTC).year}, {author}'  # noqa: A001 RUF001

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',        # Pull docstrings from source automatically.
    'sphinx.ext.viewcode',       # Add [source] links next to every item.
    'sphinx.ext.napoleon',       # Support NumPy / Google docstring styles.
    'sphinx_autodoc_typehints',  # Render PEP 484 type annotations in the docs.
]

# Keep the hosted documentation build lightweight: autodoc can document the
# llama.cpp-facing modules without installing the native llama-cpp-python wheel.
autodoc_mock_imports = ['llama_cpp']

# Treat the type annotations in signatures as the canonical type documentation
# so they are not duplicated in the parameter descriptions.
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Keep member order as defined in source rather than alphabetical.
autodoc_member_order = 'bysource'

# Include __init__ docstrings in the class entry, not as a separate heading.
autoclass_content = 'both'

# Do not skip private or special members unless they lack a docstring.
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build']

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = 'shibuya'
html_static_path = ['_static']
