#!/usr/bin/env python3
"""
generate_readme.py

Auto-generate README.md summarizing:
  - Project folder structure (listing only .py, .bat, .yaml/.yml, ignoring .git, .pytest_cache, __pycache__)
  - Quickstart usage examples
  - Dependencies from requirements.txt or environment.yml
  - Config files overview
  - CLI reference (flags & help) for entry-point scripts
  - Modules with docstrings and functions
  - Enabled features from config/features.yaml
"""

import ast
import os
import yaml
from pathlib import Path

# -------------------------
# CONFIGURATION
# -------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
INCLUDE_DIRS = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "utils",
    PROJECT_ROOT / "features",
    PROJECT_ROOT / "config"
]
EXCLUDE_FILES = {"__init__.py", "generate_readme.py"}
FEATURES_CONFIG = PROJECT_ROOT / "config" / "features.yaml"
OUTPUT_FILE = PROJECT_ROOT / "README.md"

# Folder tree filters
EXCLUDE_DIRS = {".git", ".pytest_cache", "__pycache__", "robot_env"}
INCLUDE_EXTS = {".py", ".bat", ".yaml", ".yml"}

# Quickstart commands
QUICKSTART_CMDS = [
    "python src/download_data.py --tickers data/tickers/sp500_tickers.csv",
    "python src/clean_data.py",
    "run_features_labels.bat",
    "python src/train_model.py",
]

# -------------------------
# AST PARSING HELPERS
# -------------------------
def parse_module(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    mod_doc = ast.get_docstring(tree) or ""
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            sig = f"{node.name}({', '.join(args)})"
            doc = ast.get_docstring(node) or ""
            summary = doc.strip().splitlines()[0] if doc else ""
            funcs.append((sig, summary))
    return {
        "path": path.relative_to(PROJECT_ROOT),
        "doc": mod_doc.strip(),
        "funcs": funcs
    }

# -------------------------
# Folder Structure
# -------------------------
def render_folder_tree(root: Path, max_depth=3):
    lines = []
    def recurse(p: Path, depth: int):
        if depth > max_depth:
            return
        for entry in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if entry.name in EXCLUDE_DIRS:
                continue
            prefix = "  " * depth + ("├─ " if depth else "")
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                recurse(entry, depth + 1)
            elif entry.is_file() and entry.suffix in INCLUDE_EXTS:
                lines.append(f"{prefix}{entry.name}")
    recurse(root, 0)
    return "\n".join(lines)

# -------------------------
# Dependencies
# -------------------------
def load_dependencies(root: Path):
    lines = ["## Dependencies", ""]
    req = root / "requirements.txt"
    env = root / "environment.yml"
    if req.exists():
        lines.append("**requirements.txt:**")
        for ln in req.read_text().splitlines():
            if ln.strip() and not ln.startswith("#"):
                lines.append(f"- `{ln.strip()}`")
        lines.append("")
    if env.exists():
        lines.append("**environment.yml:**")
        cfg = yaml.safe_load(env.read_text())
        for dep in cfg.get("dependencies", []):
            if isinstance(dep, str):
                lines.append(f"- `{dep}`")
            elif isinstance(dep, dict) and dep.get("pip"):
                for pip_dep in dep["pip"]:
                    lines.append(f"- `{pip_dep}`")
        lines.append("")
    if not req.exists() and not env.exists():
        lines.append("_No requirements.txt or environment.yml found._")
    return "\n".join(lines)

# -------------------------
# Config Files Overview
# -------------------------
def load_config_overview(config_dir: Path):
    lines = ["## Config Files", ""]
    for cfg_path in sorted(config_dir.glob("*.yml")) + sorted(config_dir.glob("*.yaml")):
        if cfg_path.name == FEATURES_CONFIG.name:
            desc = "pipeline feature toggles"
        elif cfg_path.name == "train_features.yaml":
            desc = "training feature toggles"
        else:
            desc = ""
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        keys = ", ".join(cfg.keys())
        lines.append(f"- **{cfg_path.relative_to(PROJECT_ROOT)}**: {desc} (keys: {keys})")
    lines.append("")
    return "\n".join(lines)

# -------------------------
# CLI Reference
# -------------------------
def extract_cli(path: Path):
    """Extract add_argument calls from a Python script."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    flags = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
           isinstance(node.func, ast.Attribute) and \
           node.func.attr == "add_argument":
            opts = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    opts.append(arg.value)
            help_txt = ""
            default = ""
            for kw in node.keywords:
                if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                    help_txt = kw.value.value
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    default = repr(kw.value.value)
            flags.append((", ".join(opts), help_txt, default))
    return flags

def render_cli_reference(src_dir: Path):
    lines = ["## CLI Reference", ""]
    for path in sorted(src_dir.glob("*.py")):
        flags = extract_cli(path)
        if not flags:
            continue
        lines.append(f"### `{path.relative_to(PROJECT_ROOT)}`")
        lines.append("")
        lines.append("| Flag | Help | Default |")
        lines.append("| ---- | ---- | ------- |")
        for opt, help_txt, default in flags:
            lines.append(f"| `{opt}` | {help_txt} | {default or '-'} |")
        lines.append("")
    return "\n".join(lines)

# -------------------------
# Quickstart
# -------------------------
def render_quickstart(cmds):
    lines = ["## Quickstart", ""]
    lines.append("```bash")
    for cmd in cmds:
        lines.append(cmd)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)

# -------------------------
# Feature Flags Loader
# -------------------------
def load_feature_flags(path: Path):
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        flags = cfg.get("features", {})
    except Exception as e:
        return f"**Error loading {path.name}:** {e}"
    lines = ["## Enabled Features", ""]
    for feat, enabled in sorted(flags.items()):
        status = "✅" if enabled else "❌"
        lines.append(f"- **{feat}**: {status}")
    lines.append("")
    return "\n".join(lines)

# -------------------------
# README Formatter
# -------------------------
def format_readme(mods, tree_str, deps_str, cfg_str, cli_str, quick_str):
    lines = []
    lines.append("# Project README")
    lines.append("")
    lines.append("*Auto-generated by `generate_readme.py`*")
    lines.append("")
    lines.append(quick_str)
    lines.append(tree_str)
    lines.append("")
    lines.append(deps_str)
    lines.append("")
    lines.append(cfg_str)
    lines.append("")
    lines.append(cli_str)
    lines.append("")
    lines.append("## Modules")
    lines.append("")
    for m in mods:
        lines.append(f"### `{m['path']}`")
        if m["doc"]:
            first_para = m["doc"].split("\n\n")[0]
            lines.append(first_para)
        if m["funcs"]:
            lines.append("**Functions:**")
            for sig, summary in m["funcs"]:
                if summary:
                    lines.append(f"- `{sig}`  \n  {summary}")
                else:
                    lines.append(f"- `{sig}`")
        lines.append("")
    lines.append("---")
    lines.append(load_feature_flags(FEATURES_CONFIG))
    lines.append("")
    return "\n".join(lines)

# -------------------------
# MAIN
# -------------------------
def main():
    # parse modules
    modules = []
    for base in INCLUDE_DIRS:
        for path in sorted(base.rglob("*.py")):
            if path.name in EXCLUDE_FILES:
                continue
            modules.append(parse_module(path))

    # build sections
    tree_str  = "```" + "\n" + render_folder_tree(PROJECT_ROOT) + "\n" + "```"
    deps_str  = load_dependencies(PROJECT_ROOT)
    cfg_str   = load_config_overview(PROJECT_ROOT / "config")
    cli_str   = render_cli_reference(PROJECT_ROOT / "src")
    quick_str = render_quickstart(QUICKSTART_CMDS)

    # assemble README
    content = format_readme(modules, tree_str, deps_str, cfg_str, cli_str, quick_str)
    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print(f"✅ Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

