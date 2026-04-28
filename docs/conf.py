# Sphinx configuration for probLM-solver
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Make the src layout visible to autodoc without requiring an editable install.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "probLM-solver"
author = "Clio"
release = "1.0.0"
copyright = f"2024, {author}"  # noqa: A001

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",        # Pull docstrings from source automatically.
    "sphinx.ext.viewcode",       # Add [source] links next to every item.
    "sphinx.ext.napoleon",       # Support NumPy / Google docstring styles.
    "sphinx_autodoc_typehints",  # Render PEP 484 type annotations in the docs.
]

# Treat the type annotations in signatures as the canonical type documentation
# so they are not duplicated in the parameter descriptions.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Keep member order as defined in source rather than alphabetical.
autodoc_member_order = "bysource"

# Include __init__ docstrings in the class entry, not as a separate heading.
autoclass_content = "both"

# Do not skip private or special members unless they lack a docstring.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "shibuya"
html_static_path = ["_static"]
