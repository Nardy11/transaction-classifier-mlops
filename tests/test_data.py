import pandas as pd
import pytest

from txclf.config import DataConfig
from txclf.data import clean, clean_description, load_raw, unique_descriptions


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WHITE HANGING HEART T-LIGHT HOLDER", "white hanging heart t light holder"),
        ("  Set/3  Cake-Tins   PANTRY ", "set 3 cake tins pantry"),
        ("Café & Crème!!", "caf cr me"),
        (None, ""),
        (float("nan"), ""),
        ("***", ""),
    ],
)
def test_clean_description(raw, expected):
    assert clean_description(raw) == expected


def _frame(rows):
    return pd.DataFrame(
        rows, columns=["UserId", "ItemDescription", "NumberOfItemsPurchased", "CostPerItem"]
    )


def test_clean_drops_duplicates_empty_and_fixes_user():
    df = _frame(
        [
            [-1, "Red Mug", 2, 1.5],
            [-1, "red  mug", 2, 1.5],  # same after cleaning -> duplicate
            [5, "???", 1, 1.0],  # empty after cleaning
            [6, None, 1, 1.0],  # missing description
        ]
    )
    out = clean(df, remove_outliers=False)
    assert out["Description"].tolist() == ["red mug"]
    assert out["UserId"].tolist() == [1]
    assert out["TotalCost"].tolist() == [3.0]


def test_clean_removes_iqr_outliers():
    rows = [[i, f"item {i}", 2, 1.0 + (i % 3) * 0.1] for i in range(40)]
    rows.append([99, "huge order", 100000, 1.0])
    out = clean(_frame(rows), remove_outliers=True)
    assert "huge order" not in out["Description"].tolist()
    assert len(out) == 40


def test_load_raw_missing_file():
    with pytest.raises(FileNotFoundError):
        load_raw(DataConfig(path="does/not/exist.csv"))


def test_unique_descriptions_counts():
    u = unique_descriptions(pd.DataFrame({"Description": ["a b", "a b", "c d"]}))
    assert dict(zip(u["Description"], u["count"], strict=True)) == {"a b": 2, "c d": 1}
