"""Command-line entry point for the benchmark project."""

import argparse

from cli_data_tool.exporter import export_text

DEFAULT_ITEMS = ["alpha", "beta", "gamma"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export sample data as text.")
    export_parser.add_argument(
        "--items",
        nargs="+",
        default=DEFAULT_ITEMS,
        help="Items to export.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "export":
        print(export_text(args.items))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
