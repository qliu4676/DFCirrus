"""Command-line interface for cirrus modeling."""

from __future__ import annotations

import argparse

from .modeling import MultiBandModeler, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dfcirrus")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a modeling configuration")
    validate.add_argument("config")

    run = commands.add_parser("run", help="run multi-band cirrus modeling")
    run.add_argument("config")
    run.add_argument("--output-dir")
    run.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        load_config(args.config, check_files=True)
        print("Configuration is valid.")
        return 0

    modeler = MultiBandModeler.from_config(args.config)
    result = modeler.run()
    output_dir = args.output_dir or modeler.config.run.output_dir
    overwrite = args.overwrite or modeler.config.run.overwrite
    result.write(output_dir, overwrite=overwrite)
    for name, color in result.colors.items():
        print(f"{name} = {color.value:.4f} +/- {color.error:.4f} mag")
    print(f"Outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
