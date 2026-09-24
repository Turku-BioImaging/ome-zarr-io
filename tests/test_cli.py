"""Tests for the `ome-zarr-io` command-line interface."""

import json
import subprocess
import sys

import pytest

from invalid_cases import make_broken_copy
from ome_zarr_io.cli import main


def drop_image_label(attrs):
    attrs["ome"].pop("image-label")


@pytest.fixture
def broken_fileset(valid_fileset, tmp_path):
    return make_broken_copy(
        valid_fileset, tmp_path / "broken.zarr", drop_image_label, label="nuclei"
    )


class TestValidate:
    def test_valid_fileset(self, valid_fileset, capsys):
        assert main(["validate", str(valid_fileset)]) == 0
        out = capsys.readouterr()
        assert "valid:     yes" in out.out
        assert "nuclei" in out.out
        assert out.err == ""

    def test_invalid_fileset(self, broken_fileset, capsys):
        assert main(["validate", str(broken_fileset)]) == 1
        out = capsys.readouterr().out
        assert "valid:     NO" in out
        assert "labels/nuclei" in out and "'image-label' is a required property" in out

    def test_json_output(self, valid_fileset, capsys):
        assert main(["validate", str(valid_fileset), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["is_valid"] is True
        assert data["spec_version"] == "0.5"
        assert data["labels"][0]["name"] == "nuclei"

    def test_json_output_for_invalid_fileset(self, broken_fileset, capsys):
        assert main(["validate", str(broken_fileset), "--json"]) == 1
        data = json.loads(capsys.readouterr().out)
        assert data["is_valid"] is False
        assert data["errors"][0]["location"] == "labels/nuclei"

    @pytest.mark.parametrize(
        "fileset_fixture,status", [("valid_fileset", 0), ("broken_fileset", 1)]
    )
    def test_quiet_prints_nothing(self, request, fileset_fixture, status, capsys):
        path = request.getfixturevalue(fileset_fixture)
        assert main(["validate", str(path), "--quiet"]) == status
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""

    def test_strict(self, valid_fileset, capsys):
        assert main(["validate", str(valid_fileset)]) == 0
        capsys.readouterr()
        assert main(["validate", str(valid_fileset), "--strict"]) == 1
        assert "valid:     NO (strict)" in capsys.readouterr().out


class TestErrors:
    def test_missing_path(self, tmp_path, capsys):
        assert main(["validate", str(tmp_path / "nope.zarr")]) == 2
        out = capsys.readouterr()
        assert out.out == ""
        assert "No OME-Zarr group found" in out.err

    def test_missing_remote_dependencies_are_explained(self, monkeypatch, capsys):
        def no_aiohttp(*args, **kwargs):
            raise ImportError('HTTPFileSystem requires "requests" and "aiohttp"')

        monkeypatch.setattr("ome_zarr_io.cli.validate", no_aiohttp)
        assert main(["validate", "https://example.org/x.zarr"]) == 2
        assert "ome-zarr-io[remote]" in capsys.readouterr().err

    def test_unreadable_target(self, monkeypatch, capsys):
        def refuse(*args, **kwargs):
            raise ConnectionError("connection refused")

        monkeypatch.setattr("ome_zarr_io.cli.validate", refuse)
        assert main(["validate", "https://example.org/x.zarr"]) == 2
        assert "connection refused" in capsys.readouterr().err

    def test_no_command_shows_help(self, capsys):
        assert main([]) == 2
        assert "validate" in capsys.readouterr().err

    def test_json_and_quiet_are_mutually_exclusive(self, valid_fileset, capsys):
        assert main(["validate", str(valid_fileset), "--json", "--quiet"]) == 2
        assert "not allowed with argument" in capsys.readouterr().err

    def test_missing_target_argument(self, capsys):
        assert main(["validate"]) == 2

    def test_version_and_help_exit_zero(self, capsys):
        assert main(["--version"]) == 0
        assert main(["validate", "--help"]) == 0


class TestModuleEntryPoint:
    """Runs the real process, to check the wiring and the actual exit status."""

    def run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "ome_zarr_io", *args], capture_output=True, text=True
        )

    def test_valid(self, valid_fileset):
        result = self.run("validate", str(valid_fileset), "--json")
        assert result.returncode == 0
        assert json.loads(result.stdout)["is_valid"] is True

    def test_invalid(self, broken_fileset):
        assert self.run("validate", str(broken_fileset), "--quiet").returncode == 1

    def test_missing(self, tmp_path):
        result = self.run("validate", str(tmp_path / "nope.zarr"))
        assert result.returncode == 2 and "error" in result.stderr
