"""Command-line interface: ``ome-zarr-io validate <path-or-url>``."""

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .report import validate

EXIT_VALID = 0
EXIT_INVALID = 1
EXIT_ERROR = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ome-zarr-io", description="Work with OME-Zarr 0.5 filesets."
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    check = subcommands.add_parser(
        "validate",
        help="validate a fileset against the OME-Zarr 0.5 schemas",
        description="Validate an OME-Zarr fileset. Exit status: 0 valid, 1 invalid, "
        "2 usage error or the target could not be read.",
    )
    check.add_argument("target", help="path or URL of the OME-Zarr group")
    check.add_argument(
        "--strict", action="store_true", help="use the stricter strict_* schemas"
    )
    output = check.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the report as JSON")
    output.add_argument(
        "--quiet", action="store_true", help="print nothing; only set the exit status"
    )
    return parser


def _error(message: str) -> int:
    print(f"ome-zarr-io: error: {message}", file=sys.stderr)
    return EXIT_ERROR


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line interface and return the process exit status."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:  # argparse exits on --help, --version and bad usage
        return int(exit_.code or 0)

    if args.command != "validate":
        parser.print_help(sys.stderr)
        return EXIT_ERROR

    try:
        report = validate(args.target, strict=args.strict)
    except FileNotFoundError as e:
        return _error(str(e))
    except ImportError as e:
        return _error(
            f"{e}. Reading remote URLs needs extra packages: "
            "pip install 'ome-zarr-io[remote]' (and s3fs for s3:// URLs)"
        )
    except OSError as e:  # includes network failures
        return _error(f"could not read '{args.target}': {e}")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        print(report)

    return EXIT_VALID if report.is_valid else EXIT_INVALID
