"""Sphinx configuration for the peacoqc documentation site."""

from __future__ import annotations

import sys
import os
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Make the editable source tree importable for autodoc.
DOCS_DIR = Path(__file__).resolve().parent
PKG_SRC = DOCS_DIR.parent.parent / "src"
sys.path.insert(0, str(PKG_SRC))

import peacoqc  # noqa: E402

# -- Project information -----------------------------------------------------
project = "peacoqc"
author = "PeacoQC Python port contributors"
copyright = "2024, PeacoQC Python port contributors"
release = peacoqc.__version__
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_param = True

# -- nbsphinx ----------------------------------------------------------------
# The vignette is pre-executed; do not re-run it during the build.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# -- intersphinx -------------------------------------------------------------
# Set SPHINX_OFFLINE=1 for deterministic builds without external inventory
# fetches (useful in CI and restricted-network environments).
if os.environ.get("SPHINX_OFFLINE") == "1":
    intersphinx_mapping = {}
else:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
        "sklearn": ("https://scikit-learn.org/stable/", None),
        "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
        "matplotlib": ("https://matplotlib.org/stable/", None),
    }

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"peacoqc {release}"

# Ignore missing references in autosummary templates that nbsphinx surfaces.
suppress_warnings = ["autosectionlabel.*"]
