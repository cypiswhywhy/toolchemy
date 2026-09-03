from pathlib import Path

import pytest
from click.testing import CliRunner

from toolchemy.agent_synergy import MARKER_BEGIN, MARKER_END, _update_file, main


def test_update_file_appends_the_block_when_the_file_has_no_markers(tmp_path):
    target = tmp_path / "AGENTS.md"
    target.write_text("# Existing guide\n", encoding="utf-8")

    action = _update_file(target)
    content = target.read_text(encoding="utf-8")

    assert action == "appended to"
    assert content.startswith("# Existing guide\n")
    assert content.count(MARKER_BEGIN) == 1
    assert content.count(MARKER_END) == 1


def test_update_file_replaces_an_existing_block_instead_of_appending_a_second(tmp_path):
    target = tmp_path / "AGENTS.md"
    target.write_text("# Existing guide\n", encoding="utf-8")
    _update_file(target)

    _update_file(target)
    content = target.read_text(encoding="utf-8")

    assert content.count(MARKER_BEGIN) == 1
    assert content.count(MARKER_END) == 1


def test_update_file_is_idempotent(tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Guide\n", encoding="utf-8")
    _update_file(target)
    after_first = target.read_text(encoding="utf-8")

    action = _update_file(target)

    assert action == "unchanged"
    assert target.read_text(encoding="utf-8") == after_first


def test_update_file_preserves_content_written_after_the_block(tmp_path):
    target = tmp_path / "AGENTS.md"
    target.write_text("# Guide\n", encoding="utf-8")
    _update_file(target)
    target.write_text(target.read_text(encoding="utf-8") + "\n## Trailing section\n", encoding="utf-8")

    _update_file(target)
    content = target.read_text(encoding="utf-8")

    assert "# Guide" in content
    assert "## Trailing section" in content
    assert content.count(MARKER_BEGIN) == 1


def test_update_file_creates_nothing_but_still_writes_when_the_file_is_empty(tmp_path):
    target = tmp_path / "AGENTS.md"
    target.write_text("", encoding="utf-8")

    action = _update_file(target)

    assert action == "appended to"
    assert target.read_text(encoding="utf-8").startswith(MARKER_BEGIN)


def test_main_exits_non_zero_when_no_target_file_exists(tmp_path):
    result = CliRunner().invoke(main, ["--path", str(tmp_path)])

    assert result.exit_code == 1
    assert "No AGENTS.md or CLAUDE.md found" in result.output


@pytest.mark.parametrize("filenames", [["AGENTS.md"], ["CLAUDE.md"], ["AGENTS.md", "CLAUDE.md"]])
def test_main_updates_every_target_file_present(tmp_path, filenames):
    for name in filenames:
        (tmp_path / name).write_text("# Guide\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["--path", str(tmp_path)])

    assert result.exit_code == 0
    for name in filenames:
        assert MARKER_BEGIN in (tmp_path / name).read_text(encoding="utf-8")
    assert Path(tmp_path).exists()
