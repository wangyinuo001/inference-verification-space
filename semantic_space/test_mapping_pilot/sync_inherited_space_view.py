"""Add reviewed property rows to the explorer's on-demand evidence bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA = HERE.parent / "explorer_extra.json"
DATA = HERE / "inherited-property-crosswalk.tsv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def keys(rows: list[dict[str, str]]) -> list[tuple[str, str, str, str]]:
    return [(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in rows]


def main() -> dict[str, int]:
    inherited = read(DATA)
    reviewed = read(HERE / "confirmed-property-crosswalk.tsv")
    reviewed += read(HERE / "direct-property-crosswalk.tsv")
    inherited_keys, reviewed_keys = keys(inherited), keys(reviewed)
    if (len(inherited_keys) != len(set(inherited_keys)) or
            len(reviewed_keys) != len(set(reviewed_keys)) or
            set(inherited_keys) & set(reviewed_keys)):
        raise ValueError("Duplicate reviewed crosswalk keys")
    bundle = json.loads(EXTRA.read_text(encoding="utf-8"))
    bundle["inheritedProperty"] = inherited
    bundle["reviewedProperty"] = reviewed
    EXTRA.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    result = {"reviewed_inherited": len(inherited), "reviewed_direct": len(reviewed),
              "bytes": EXTRA.stat().st_size}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
