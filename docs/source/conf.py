project = "qafm"
author = "Jannik Evers, Philipp Rahe"
copyright = "2026, Jannik Evers and Philipp Rahe"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_design",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

#html_theme = "sphinx_rtd_theme" 
html_theme = "furo"

# Optional: if you keep any .rst files, both are supported.
# Otherwise this makes .md the default:
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

