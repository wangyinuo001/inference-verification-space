"""Reconcile the full classification with the indexed source and reviewed space."""
import csv
import json
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
    checked = rows(HERE/'reviewed-round10-crosswalk.tsv')
    receipts = json.loads((HERE/'reviewed-round10-receipts.json').read_text())
    assert len(checked) == len(receipts)
    assert {tuple(r[k] for k in KEY) for r in checked} == {tuple(r['key']) for r in receipts}
    graph = (HERE.parent/'space_view.html').read_text()
    for row in checked:
        assert f'data-unit="{row["unit_id"]}"' in graph
    assert all(r['source_sha256'] and r['oracle_source_sha256'] for r in receipts)
