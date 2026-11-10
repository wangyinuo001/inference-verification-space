"""Find where inherited Python test methods are actually defined.

This resolves local and explicit imported base classes without importing or
running the test suite. A resolved method is only a source location, not a
claim that the test verifies a particular property.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE = HERE.parent
REPOS = SPACE.parent / "third_party" / "reference_repos"
ROOTS = {"vllm": REPOS / "vllm-20261105", "sglang": REPOS / "sglang-v0.5.14-complete"}
OUTPUT = HERE / "inherited-method-sources.tsv"
FIELDS = ("snapshot", "path", "test_name", "line", "status", "source_path",
          "source_line", "source_class", "source_method", "candidate_count")


class Resolver:
    def __init__(self, root: Path):
        self.root = root
        self.cache: dict[Path, tuple[dict[str, ast.ClassDef], dict[str, tuple[Path, str]], dict[str, list[str]]]] = {}

    def module_path(self, module: str, current: Path, level: int = 0) -> Path | None:
        if level:
            base = current.parent
            for _ in range(level - 1):
                base = base.parent
            path = base.joinpath(*(module.split(".") if module else []))
            candidates = [path.with_suffix(".py"), path / "__init__.py"]
        else:
            parts = module.split(".")
            candidates = []
            # Some test suites explicitly add their own directory to sys.path
            # and import a nearby _common.py module.
            if parts[0].startswith(("_", "test_")):
                local = current.parent.joinpath(*parts)
                candidates.extend([local.with_suffix(".py"), local / "__init__.py"])
            for prefix in (self.root, self.root / "python"):
                path = prefix.joinpath(*parts)
                candidates.extend([path.with_suffix(".py"), path / "__init__.py"])
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def parse(self, path: Path) -> tuple[dict[str, ast.ClassDef], dict[str, tuple[Path, str]], dict[str, list[str]]]:
        if path in self.cache:
            return self.cache[path]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, UnicodeError):
            self.cache[path] = ({}, {}, {})
            return self.cache[path]
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        imports: dict[str, tuple[Path, str]] = {}
        groups: dict[str, list[str]] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, (ast.Tuple, ast.List)) and all(isinstance(item, ast.Name) for item in node.value.elts):
                    groups[node.targets[0].id] = [item.id for item in node.value.elts]
            if isinstance(node, ast.ImportFrom):
                imported_path = self.module_path(node.module or "", path, node.level)
                if imported_path:
                    for alias in node.names:
                        if alias.name != "*":
                            imports[alias.asname or alias.name] = (imported_path, alias.name)
        self.cache[path] = (classes, imports, groups)
        return self.cache[path]

    def find(self, path: Path, cls: str, method: str,
             visited: set[tuple[Path, str]] | None = None) -> list[tuple[Path, str, int]]:
        visited = set() if visited is None else visited
        key = (path, cls)
        if key in visited:
            return []
        visited.add(key)
        classes, imports, groups = self.parse(path)
        node = classes.get(cls)
        if node is None:
            return []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method:
                return [(path, cls, child.lineno)]
        results: list[tuple[Path, str, int]] = []
        for base in node.bases:
            expanded = [base]
            if isinstance(base, ast.Starred) and isinstance(base.value, ast.Name):
                expanded = [ast.Name(id=name) for name in groups.get(base.value.id, [])]
            for item in expanded:
                results.extend(self.find_base(path, item, method, visited, classes, imports))
        return list(dict.fromkeys(results))

    def find_base(self, path: Path, base: ast.expr, method: str,
                  visited: set[tuple[Path, str]], classes: dict[str, ast.ClassDef],
                  imports: dict[str, tuple[Path, str]]) -> list[tuple[Path, str, int]]:
        while isinstance(base, ast.Subscript):
            base = base.value
        if not isinstance(base, ast.Name):
            return []
        name = base.id
        if name in classes:
            return self.find(path, name, method, visited)
        if name in imports:
            base_path, base_name = imports[name]
            return self.find(base_path, base_name, method, visited)
        return []


def main() -> dict[str, int]:
    missing = [name for name, root in ROOTS.items() if not (root / ("vllm" if name == "vllm" else "python/sglang")).is_dir()]
    if missing:
        raise FileNotFoundError(f"Full source checkouts required for: {', '.join(missing)}")
    with (SPACE / "master-review-index.tsv").open(encoding="utf-8", newline="") as f:
        entries = [row for row in csv.DictReader(f, delimiter="\t")
                   if row["entry_kind"] == "python_inherited_test_candidate"]
    resolvers = {name: Resolver(root) for name, root in ROOTS.items()}
    rows = []
    for entry in entries:
        snapshot = entry["snapshot"]
        resolver = resolvers[snapshot]
        cls, _, method = entry["test_name"].rpartition("::")
        source = resolver.root / entry["path"]
        matches = resolver.find(source, cls, method) if cls else []
        status = "resolved_unique" if len(matches) == 1 else "resolved_multiple" if matches else "unresolved"
        primary = matches[0] if matches else None
        rows.append({
            "snapshot": snapshot, "path": entry["path"], "test_name": entry["test_name"],
            "line": entry["line"], "status": status,
            "source_path": str(primary[0].relative_to(resolver.root)) if primary else "",
            "source_line": str(primary[2]) if primary else "",
            "source_class": primary[1] if primary else "", "source_method": method,
            "candidate_count": str(len(matches)),
        })
    assert len(rows) == len({(row["snapshot"], row["path"], row["test_name"], row["line"]) for row in rows})
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(Counter(row["status"] for row in rows))
    print(json.dumps({"total": len(rows), **summary}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
