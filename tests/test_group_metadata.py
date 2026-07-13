from scat.grouping_util import build_group_metadata


def test_build_metadata_columns():
    df, group_by = build_group_metadata({"a.tif": "control", "b.tif": "treated", "c.tif": None})
    assert group_by == ["group"]
    assert set(df.columns) == {"filename", "group"}
    row = df[df.filename == "c.tif"].iloc[0]
    assert row["group"] == "ungrouped"
