"""Build documentation without importing warpkep or its GPU dependencies."""

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "pyproject.toml").open("rb") as metadata:
    release = tomllib.load(metadata)["project"]["version"]

project = "warpkep"
author = "Benjamin von Plehn"
copyright = "2026, Benjamin von Plehn"
version = release
language = "en"

extensions = ["myst_parser", "sphinx.ext.mathjax"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "requirements.txt"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
nitpicky = True

html_theme = "furo"
html_title = f"warpkep {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/benplehn/warpkep/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#145c73",
        "color-brand-content": "#145c73",
    },
    "dark_css_variables": {
        "color-brand-primary": "#70d6e4",
        "color-brand-content": "#70d6e4",
    },
}

linkcheck_ignore = [r"http://localhost(?::\d+)?(?:/.*)?$"]
