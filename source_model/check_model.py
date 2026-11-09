"""Validate space_model.json and answer change-scope queries.

    python check_model.py                       # validate + summary
    python check_model.py --changed v:v1/core/block_pool.py s:mem_cache/radix_cache.py

Validation: schema values, dependency closure and acyclicity, condition and
perturbation references, and that every cited source range exists in the
pinned snapshots (found under third_party/reference_repos, or set VERIFICATION_REFERENCE_REPOS).
The change query lists units whose cited sites lie in the changed files, then
every unit that depends on them (transitively). It is a scope query over the
space, not a test-selection method.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    return json.loads((HERE / "space_model.json").read_text())


def reference_dir():
    env = os.environ.get("VERIFICATION_REFERENCE_REPOS")
    if env:
        return Path(env)
    for parent in HERE.parents:
        for rel in ("third_party/reference_repos", "experiments/third_party/reference_repos"):
            if (parent / rel / "vllm-20261105").is_dir():
                return parent / rel
    sys.exit("reference snapshots not found; set VERIFICATION_REFERENCE_REPOS")


def snapshot_roots(model):
    base = reference_dir()
    return {key[0]: base / snap["path"] for key, snap in model["snapshots"].items()}


def parse_site(site):
    prefix, path, lines = site.split(":")
    a, _, b = lines.partition("-")
    return prefix, path, int(a), int(b or a)


def validate(model):
    errors = []
    axes = model["axes"]
    units = {u["id"]: u for u in model["units"]}
    if len(units) != len(model["units"]):
        errors.append("duplicate unit ids")
    roots = snapshot_roots(model)
    line_cache = {}
    for u in model["units"]:
        for field in ("stage", "kind", "boundary"):
            if u[field] not in axes[field]:
                errors.append(f"{u['id']}: bad {field} {u[field]}")
        for c in u["conditions"]:
            if c not in model["conditions"]:
                errors.append(f"{u['id']}: unknown condition {c}")
        for d in u["depends_on"]:
            if d not in units:
                errors.append(f"{u['id']}: unknown dependency {d}")
        if u["perturbation"] and u["perturbation"] not in model["perturbations"]:
            errors.append(f"{u['id']}: unknown perturbation {u['perturbation']}")
        if u["kind"] == "equivalence" and not u["perturbation"]:
            errors.append(f"{u['id']}: equivalence unit without perturbation")
        if u["status"] == "common" and not (u["sites"].get("vllm") and u["sites"].get("sglang")):
            errors.append(f"{u['id']}: common unit must cite both frameworks")
        for fw, sites in u["sites"].items():
            for s in sites:
                prefix, path, a, b = parse_site(s)
                if prefix != fw[0]:
                    errors.append(f"{u['id']}: {s} listed under {fw}")
                f = roots[prefix] / path
                if f not in line_cache:
                    line_cache[f] = sum(1 for _ in f.open()) if f.exists() else -1
                n = line_cache[f]
                if n < 0:
                    errors.append(f"{u['id']}: missing file {s}")
                elif not (1 <= a <= b <= n):
                    errors.append(f"{u['id']}: {s} outside 1..{n}")
    for c in model["constraints"]:
        prefix, path, a, b = parse_site(c["site"])
        f = roots[prefix] / path
        if not f.exists() or b > sum(1 for _ in f.open()):
            errors.append(f"constraint {c['id']}: bad site {c['site']}")
    # acyclicity
    state = {}

    def visit(n):
        if state.get(n) == 1:
            errors.append(f"dependency cycle at {n}")
            return
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in units[n]["depends_on"]:
            if d in units:
                visit(d)
        state[n] = 2

    for n in units:
        visit(n)
    order = list(axes["boundary"])
    for u in model["units"]:
        for d in u["depends_on"]:
            if d in units and order.index(units[d]["boundary"]) > order.index(u["boundary"]):
                errors.append(f"{u['id']}: depends on higher-boundary unit {d}")
        if u["depends_on"] and order.index(u["boundary"]) >= order.index("B3") and not u["composition_gap"]:
            if u["kind"] == "equivalence":
                errors.append(f"{u['id']}: request-level equivalence without stated composition gap")
    return errors


def summary(model):
    units = model["units"]
    print(f"units={len(units)} conditions={len(model['conditions'])} "
          f"perturbations={len(model['perturbations'])} constraints={len(model['constraints'])}")
    print("status:", dict(Counter(u["status"] for u in units)))
    print("with framework divergence:", sum(1 for u in units if u["divergence"]))
    print("with composition gap:", sum(1 for u in units if u["composition_gap"]))
    print("sites:", sum(len(s) for u in units for s in u["sites"].values()))
    bounds = list(model["axes"]["boundary"])
    stages = list(model["axes"]["stage"])
    grid = Counter((u["stage"], u["boundary"]) for u in units)
    print("\nstage x boundary")
    print(f"{'':14}" + "".join(f"{b:>4}" for b in bounds))
    for s in stages:
        print(f"{s:14}" + "".join(f"{grid[(s, b)] or '.':>4}" for b in bounds))
    kinds = Counter((u["kind"], u["boundary"]) for u in units)
    print("\nkind x boundary")
    for k in model["axes"]["kind"]:
        print(f"{k:14}" + "".join(f"{kinds[(k, b)] or '.':>4}" for b in bounds))


def changed_scope(model, changed):
    units = {u["id"]: u for u in model["units"]}
    rdeps = defaultdict(set)
    for u in model["units"]:
        for d in u["depends_on"]:
            rdeps[d].add(u["id"])
    direct = set()
    for u in model["units"]:
        for sites in u["sites"].values():
            for s in sites:
                prefix, path, _, _ = parse_site(s)
                if f"{prefix}:{path}" in changed:
                    direct.add(u["id"])
    closure, stack = set(direct), list(direct)
    while stack:
        for r in rdeps[stack.pop()]:
            if r not in closure:
                closure.add(r)
                stack.append(r)
    order = list(model["axes"]["boundary"])
    for uid in sorted(closure, key=lambda i: (order.index(units[i]["boundary"]), i)):
        u = units[uid]
        tag = "direct " if uid in direct else "upward "
        print(f"{tag}{u['boundary']} {uid:24} {u['kind']:12} conditions={','.join(u['conditions'])}")
    unmapped = [c for c in changed if not any(
        f"{parse_site(s)[0]}:{parse_site(s)[1]}" == c
        for u in model["units"] for ss in u["sites"].values() for s in ss)]
    for c in unmapped:
        print(f"unmapped {c}: no unit cites this file; scope is unknown, not empty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed", nargs="*")
    args = ap.parse_args()
    model = load()
    errors = validate(model)
    for e in errors:
        print("ERROR", e)
    if errors:
        sys.exit(1)
    if args.changed:
        changed_scope(model, set(args.changed))
    else:
        summary(model)


if __name__ == "__main__":
    main()
