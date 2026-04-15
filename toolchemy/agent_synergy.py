"""`agent-synergy` console command.

Run inside any project that depends on toolchemy. If an AGENTS.md or CLAUDE.md
file exists in the current working directory, insert (between markers) a short,
strongly-worded pointer to the toolchemy capability manifest shipped inside the
installed package. The pointer includes the manifest's resolved absolute path
so an agent can read it directly with a single file-read, without running any
helper command. Idempotent — re-running refreshes the resolved path.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import click

MARKER_BEGIN = "<!-- toolchemy:agent-synergy:begin -->"
MARKER_END = "<!-- toolchemy:agent-synergy:end -->"
TARGET_FILENAMES = ("AGENTS.md", "CLAUDE.md")


def _manifest_path() -> str:
    try:
        resource = files("toolchemy").joinpath("AGENTS_MANIFEST.md")
        with as_file(resource) as concrete:
            return str(Path(concrete).resolve())
    except Exception:  # noqa: BLE001
        return "<toolchemy package>/AGENTS_MANIFEST.md"


def _pointer_block() -> str:
    manifest = _manifest_path()
    body = "\n".join([
        "## Reusing `toolchemy` — MANDATORY",
        "",
        "This project depends on the `toolchemy` package, which ships a capability",
        "manifest listing every reusable symbol it exposes (helpers, clients,",
        "trackers, cache backends, vision/nlp utilities, etc.).",
        "",
        f"**You MUST read this file once at the start of the session:** `{manifest}`",
        "",
        "Then, before writing ANY new utility, helper, client, wrapper, or",
        "integration, check whether the manifest already lists a symbol that covers",
        "the need. If it does, import and reuse it instead of reimplementing. Do not",
        "assume a capability is missing just because it is not mentioned in this",
        "file — the manifest is the authoritative, up-to-date inventory.",
    ])
    return "\n".join([MARKER_BEGIN, body, MARKER_END, ""])


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
