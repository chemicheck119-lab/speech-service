#!/usr/bin/env python3
"""Export or verify the deterministic bounded Speech API contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemicheck119_speech.api import create_app


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "contracts" / "speech-api.openapi.json"


def rendered_contract() -> str:
    payload = create_app().openapi()
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rendered = rendered_contract()
    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            raise SystemExit("Speech OpenAPI snapshot이 현재 코드와 다릅니다.")
        print("Speech OpenAPI snapshot이 현재 코드와 일치합니다.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
