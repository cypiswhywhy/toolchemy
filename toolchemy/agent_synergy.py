"""`agent-synergy` console command.

Run inside any project that depends on toolchemy. If an AGENTS.md or CLAUDE.md
file exists in the current working directory, append a pointer block telling
coding agents to consult toolchemy's capability manifest before reimplementing
utilities. Idempotent — re-running will not duplicate the block.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import click

MARKER_BEGIN = "<!-- toolchemy:agent-synergy:begin -->"
MARKER_END = "<!-- toolchemy:agent-synergy:end -->"
TARGET_FILENAMES = ("AGENTS.md", "CLAUDE.md")


def _manifest_resource_path() -> str:
    try:
        path = files("toolchemy").joinpath("AGENTS_MANIFEST.md")
        return str(path)
    except Exception:  # noqa: BLE001
        return "<toolchemy package>/AGENTS_MANIFEST.md"


def _pointer_block() -> str:
    manifest = _manifest_resource_path()
    return "\n".join([
        MARKER_BEGIN,
        "## Reusing `toolchemy`",
        "",
        "This project depends on the `toolchemy` package. Before implementing any",
        "utility, helper, or integration, consult the capability manifest shipped",
        "inside the package and prefer reusing a listed symbol over writing new code:",
        "",
        f"- Manifest path: `{manifest}`",
        "- Resolve at runtime with:",
        "  `python -c \"from importlib.resources import files; print(files('toolchemy').joinpath('AGENTS_MANIFEST.md'))\"`",
        "",
        "The manifest is regenerated on each `toolchemy` release, so treat it as the",
        "authoritative, up-to-date inventory of what is available.",
        MARKER_END,
        "",
    ])


def _update_file(path: Path) -> str:
    block = _pointer_block()
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER_BEGIN in original and MARKER_END in original:
        pre, _, rest = original.partition(MARKER_BEGIN)
        _, _, post = rest.partition(MARKER_END)
        post = post.lstrip("\n")
        new_text = pre.rstrip() + "\n\n" + block + ("\n" + post if post else "")
        action = "updated"
    else:
        sep = "" if original.endswith("\n\n") or not original else ("\n" if original.endswith("\n") else "\n\n")
        new_text = original + sep + block
        action = "appended to"
    if new_text == original:
        return "unchanged"
    path.write_text(new_text, encoding="utf-8")
    return action


@click.command()
@click.option(
    "--path",
    "project_path",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, exists=True),
    help="Project directory to scan. Defaults to the current working directory.",
)
def main(project_path: str) -> None:
    """Extend AGENTS.md and/or CLAUDE.md with a pointer to the toolchemy manifest."""
    root = Path(project_path).resolve()
    found = [root / name for name in TARGET_FILENAMES if (root / name).exists()]
    if not found:
        click.echo(
            f"No AGENTS.md or CLAUDE.md found in {root}. "
            "Create one first, then re-run `agent-synergy`."
        )
        raise SystemExit(1)
    for target in found:
        action = _update_file(target)
        click.echo(f"{action}: {target}")


if __name__ == "__main__":
    main()
