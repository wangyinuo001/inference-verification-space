"""Discover validated review batches; proposal files are never included."""
import re
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r'reviewed-round(\d+)-crosswalk\.tsv$')


def batch_files(root=HERE):
    found=[]
    for path in root.glob('reviewed-round*-crosswalk.tsv'):
        match=PATTERN.fullmatch(path.name)
        if not match:
            continue
        receipt=path.with_name(path.name.replace('-crosswalk.tsv','-receipts.json'))
        if not receipt.is_file():
            raise ValueError(f'Missing source receipts for {path.name}')
        found.append((int(match[1]),path))
    return [path for _,path in sorted(found)]


def source_review_corrections(root=HERE):
    path = root / 'classification-corrections.json'
    if not path.exists():
        return {}
    result = {}
    for correction in json.loads(path.read_text()):
        key = tuple(str(correction[k]) for k in ('snapshot', 'path', 'test_name', 'line'))
        if key in result:
            raise ValueError(f'Duplicate classification correction: {key}')
        if not correction.get('reason') or not correction.get('new_unit_id'):
            raise ValueError(f'Incomplete classification correction: {key}')
        result[key] = correction
    return result


def apply_source_review_correction(row, corrections):
    key = tuple(str(row[k]) for k in ('snapshot', 'path', 'test_name', 'line'))
    correction = corrections.get(key)
    if correction is None:
        return row, None
    if row['unit_id'] != correction['old_unit_id']:
        raise ValueError(f'Correction source no longer matches: {key}')
    updated = dict(row, unit_id=correction['new_unit_id'])
    for field in ('property', 'conditions', 'oracle_source'):
        if field in correction:
            updated[field] = correction[field]
    return updated, correction
