# Configuration file for Sphinx documentation

import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

project = "FaceCompare"
author = "griffin"
copyright = "2025, griffin"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]

# Source file configurations
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Cache settings for better performance
cache_path = "_build/doctrees"

# Additional settings for better documentation
add_module_names = False
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = True
