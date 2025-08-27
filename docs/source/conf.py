# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "Inspect WandB"
copyright = "2025, Inspect WandB"
author = "Inspect WandB"


# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# MyST Parser configuration - enables macros and cross-references
myst_enable_extensions = [
    "substitution",  # Enables {{ macro_name }} syntax
    "colon_fence",   # Enables {ref}`label` and {doc}`filename` links
]

# Global substitutions (macros) - use {{ macro_name }} in .md files
# Update these values once to change them everywhere in docs
myst_substitutions = {
    "repo_url": "https://github.com/DanielPolatajko/inspect_wandb.git",
    "install_basic": "pip install git+https://github.com/DanielPolatajko/inspect_wandb.git",
    "install_weave": 'pip install inspect_wandb @ "git+https://github.com/DanielPolatajko/inspect_wandb.git[weave]"',
    "install_full": 'pip install inspect_wandb @ "git+https://github.com/DanielPolatajko/inspect_wandb.git[weave,viz]"',
}

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]