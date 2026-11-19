"""Validate a completed source-review batch against all earlier batches."""
import argparse
from build_round9_crosswalk import HERE, main as validate
from reviewed_batches import batch_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=int)
    args = parser.parse_args()
    batch = HERE / f"round{args.number}_work"
    prior = tuple(p.name for p in batch_files() if int(p.name.split("round")[1].split("-")[0]) < args.number)
    validate(batch=batch, label=f"round{args.number}", names=[p.name for p in sorted(batch.iterdir()) if p.is_dir()], prior=prior)
