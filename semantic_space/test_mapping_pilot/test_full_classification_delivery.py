"""Reconcile the full classification with the indexed source and reviewed space."""
import csv
import json
from reviewed_batches import batch_files
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = ('snapshot', 'path', 'test_name', 'line')


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def test_all_entries_have_a_disposition_without_changing_reviewed_claims():
    baseline = {tuple(r[k] for k in KEY): r for r in rows(HERE/'full_delivery/all-entries.tsv')}
    classified = rows(HERE/'full_classification/all-entries.tsv')
    assert len(classified) == len(baseline) == 31968
    assert {tuple(r[k] for k in KEY) for r in classified} == set(baseline)
    for row in classified:
        old = baseline[tuple(row[k] for k in KEY)]
        assert row['source_sha256'] == old['source_sha256']
        assert row['reviewed_units'] == old['unit_ids']
        assert (row['disposition'] == 'reviewed_property') == (old['review_status'] == 'property_reviewed')


def test_new_reviewed_mappings_have_registered_visible_units_and_receipts():
    graph = (HERE.parent/'space_view.html').read_text()
    for path in batch_files():
        checked = rows(path)
        receipts = json.loads(path.with_name(path.name.replace('-crosswalk.tsv', '-receipts.json')).read_text())
        assert len(checked) == len(receipts)
        assert {tuple(r[k] for k in KEY) for r in checked} == {tuple(r['key']) for r in receipts}
        for row in checked:
            for unit in row["unit_id"].split(";"):
                assert f'data-unit="{unit}"' in graph
        assert all(r['source_sha256'] and r['oracle_source_sha256'] for r in receipts)


def test_batch_discovery_excludes_proposals_and_requires_receipts(tmp_path):
    import pytest
    (tmp_path/'proposed-round99-crosswalk.tsv').write_text('proposal')
    assert batch_files(tmp_path) == []
    checked = tmp_path/'reviewed-round12-crosswalk.tsv'
    checked.write_text('reviewed')
    with pytest.raises(ValueError, match='Missing source receipts'):
        batch_files(tmp_path)
    (tmp_path/'reviewed-round12-receipts.json').write_text('[]')
    assert batch_files(tmp_path) == [checked]


def test_corrections_preserve_original_mapping_and_reject_stale_source():
    import pytest
    from reviewed_batches import source_review_corrections, apply_source_review_correction
    corrections = source_review_corrections(HERE)
    for path in batch_files():
        checked = {tuple(r[k] for k in KEY): r for r in rows(path)}
        receipts = json.loads(path.with_name(path.name.replace('-crosswalk.tsv', '-receipts.json')).read_text())
        for receipt in receipts:
            correction = receipt.get('classification_correction')
            if not correction:
                continue
            key = tuple(str(v) for v in receipt['key'])
            original = next(r for r in rows(HERE/receipt['review_file']) if tuple(r[k] for k in KEY) == key)
            assert original['unit_id'] == correction['old_unit_id']
            updated, evidence = apply_source_review_correction(original, corrections)
            assert updated == checked[key]
            assert evidence == correction
            with pytest.raises(ValueError, match='no longer matches'):
                apply_source_review_correction(dict(original, unit_id='unexpected.reclassification'), corrections)
