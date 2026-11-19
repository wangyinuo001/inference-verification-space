"""Validate the next parallel source-review batch without changing older reviews."""
from build_round9_crosswalk import HERE, main as validate

if __name__ == "__main__":
    validate(batch=HERE / "round10_work", label="round10",
             names=[p.name for p in sorted((HERE / "round10_work").iterdir()) if p.is_dir()],
             prior=("reviewed-round9-crosswalk.tsv",))
