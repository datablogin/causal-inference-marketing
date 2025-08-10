"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# Add paths to the Python modules
sys.path.insert(0, os.path.abspath("../../libs/causal_inference"))
sys.path.insert(0, os.path.abspath("../../shared"))
sys.path.insert(0, os.path.abspath("../../services"))

project = "Causal Inference Marketing Tools"
copyright = "2024, Causal Inference Marketing Team"
author = "Causal Inference Marketing Team"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
