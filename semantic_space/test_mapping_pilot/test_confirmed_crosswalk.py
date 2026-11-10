"""Check that the source-reviewed crosswalk stays narrow and reproducible."""

import csv
from pathlib import Path

from build_confirmed_crosswalk import main, target

HERE = Path(__file__).resolve().parent


def test_confirmed_crosswalk():
    main()
    with (HERE / "confirmed-property-crosswalk.tsv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 204
    assert all(row["property"] and row["oracle_source"] and row["unit_id"] for row in rows)
    assert len({(row["snapshot"], row["path"], row["test_name"], row["line"]) for row in rows}) == len(rows)
    assert not any("test_network_address.py" in row["path"] for row in rows)
    assert not any("test_video.py" in row["path"] for row in rows)
    assert sum("test_deferred_block_free.py" in row["path"] for row in rows) == 12
    assert next(row for row in rows if row["test_name"] == "TestHunyuanDetectorStreaming::test_streaming_typed_args_coerced")["unit_id"] == "output.tool_argument_type_binding"
    assert target({"path": "test/registered/unit/utils/test_auth.py", "test_name": "TestAuthDecision::test_frozen"}) is None
    assert target({"path": "tests/v1/core/test_deferred_block_free.py", "test_name": "test_gate_enabled_for_async_consumer"}) is None
    assert target({"path": "test/registered/unit/function_call/test_hunyuan_detector.py", "test_name": "TestHunyuanDetectorStructureInfo::test_structure_info_content"}) is None
