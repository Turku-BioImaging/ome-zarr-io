"""Run the code examples in the README and the documentation.

Every page is run on its own, in an empty temporary directory, so each page's
examples must work without help from other pages. Within a page the Python blocks
run in order and share one namespace, like cells in a notebook, and the shell
blocks run afterwards against the files the Python blocks wrote.

* ``python`` blocks are executed. Warnings are errors here, as in the rest of the
  test suite, so an example that triggers a warning fails.
* ``bash`` blocks: only the lines that start with ``ome-zarr-io`` are run, as
  ``python -m ome_zarr_io``. The exit status may be 0 (valid) or 1 (invalid) but not
  2 (usage error), so a wrong flag or subcommand fails. Other lines, such as
  ``pip install``, are not run.
* To leave out a block that cannot run (for example one that needs a network),
  put ``<!-- skip-test -->`` on the line directly above it.
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_MARKER = "<!-- skip-test -->"
FENCE = re.compile(r"^```(python|bash)\s*$")


class Snippet(NamedTuple):
    language: str
    line: int  # 1-based line of the opening fence
    code: str


def find_pages() -> List[Path]:
    pages = [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md"))
    return [p for p in pages if p.exists() and "_build" not in p.parts]


def extract_snippets(path: Path) -> List[Snippet]:
    """The runnable fenced code blocks of a Markdown file, in document order."""
    lines = path.read_text(encoding="utf-8").split("\n")
    snippets: List[Snippet] = []
    i = 0
    while i < len(lines):
        match = FENCE.match(lines[i])
        if not match:
            i += 1
            continue
        start = i
        i += 1
        body = []
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i])
            i += 1
        skipped = start > 0 and lines[start - 1].strip() == SKIP_MARKER
        if not skipped:
            snippets.append(Snippet(match.group(1), start + 1, "\n".join(body)))
        i += 1  # the closing fence
    return snippets


def shell_commands(code: str) -> List[List[str]]:
    """The ``ome-zarr-io`` commands in a bash block, as argument lists."""
    commands = []
    for raw in code.split("\n"):
        for part in raw.split("&&"):
            part = part.split("#")[0].strip()
            if part.startswith("ome-zarr-io"):
                commands.append(shlex.split(part)[1:])
    return commands


PAGES = [p for p in find_pages() if extract_snippets(p)]


def page_id(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_snippets_are_found():
    """Guards against the extraction silently finding nothing."""
    found = {page_id(p): extract_snippets(p) for p in PAGES}
    python_blocks = sum(1 for s in found.values() for x in s if x.language == "python")
    assert python_blocks >= 8, found
    for page in ("README.md", "docs/index.md", "docs/guides/writing-images.md",
                 "docs/guides/plates.md", "docs/guides/reading.md"):
        assert page in found, f"{page} has no runnable snippets"


def test_extraction_handles_skip_marker_and_languages(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        "text\n\n"
        "```python\nx = 1\n```\n\n"
        f"{SKIP_MARKER}\n```python\nraise SystemExit\n```\n\n"
        "```bash\nome-zarr-io validate a.zarr  # comment\n```\n\n"
        "```{toctree}\nnot-code\n```\n\n"
        "```text\nignored\n```\n"
    )
    snippets = extract_snippets(page)
    assert [(s.language, s.line) for s in snippets] == [("python", 3), ("bash", 12)]
    assert shell_commands("pip install x\nome-zarr-io a --strict --quiet && echo ok") == [
        ["a", "--strict", "--quiet"]
    ]


@pytest.mark.skipif(not (ROOT / "docs").is_dir(), reason="documentation not available")
@pytest.mark.parametrize("page", PAGES, ids=page_id)
def test_page_snippets_run(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__docs_snippet__"}
    shell: List[Snippet] = []

    for snippet in extract_snippets(page):
        if snippet.language == "bash":
            shell.append(snippet)
            continue
        # Pad with blank lines so a traceback shows the line number in the Markdown file.
        source = "\n" * snippet.line + snippet.code
        exec(compile(source, str(page), "exec"), namespace)

    for snippet in shell:
        for args in shell_commands(snippet.code):
            result = subprocess.run(
                [sys.executable, "-m", "ome_zarr_io", *args],
                capture_output=True,
                text=True,
                cwd=tmp_path,
            )
            assert result.returncode in (0, 1), (
                f"{page_id(page)}:{snippet.line}: `ome-zarr-io {' '.join(args)}` "
                f"exited with {result.returncode}\n{result.stdout}\n{result.stderr}"
            )
