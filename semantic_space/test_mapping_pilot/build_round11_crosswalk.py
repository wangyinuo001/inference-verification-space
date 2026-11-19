"""Check source-reviewed contracts and their pinned evidence before publishing."""
from build_round9_crosswalk import HERE, main as validate

if __name__ == "__main__":
    validate(batch=HERE / "round11_work", label="round11",
             names=[p.name for p in sorted((HERE / "round11_work").iterdir()) if p.is_dir()],
             prior=("reviewed-round9-crosswalk.tsv", "reviewed-round10-crosswalk.tsv"))
