import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data  # noqa: E402


def fake_raw(monkeypatch, rows):
    df = pd.DataFrame(rows, columns=["text", "dialect"])
    df["label"] = df["dialect"].map(data.DIALECTS)
    monkeypatch.setattr(data, "load_raw", lambda: df)


def test_drops_short_conflicting_and_duplicate_tweets(monkeypatch):
    fake_raw(monkeypatch, [
        ("انا مش فاهم حاجة خالص", "EG"),
        ("انا مش فاهم حاجة خالص", "EG"),        # duplicate, keep one
        ("@someone انا مش فاهم حاجة خالص", "EG"),  # same after removing the mention
        ("شو عم تعمل هلق", "LB"),
        ("كيفك", "LB"),                            # too short
        ("الله يسلمك يا خوي", "LY"),
        ("الله يسلمك يا خوي", "SD"),              # same text, two dialects: drop both
    ])
    df = data.load_clean()
    assert sorted(df["label"]) == ["Egyptian", "Lebanese"]


def test_split_sizes_and_no_overlap(monkeypatch):
    rows = [(f"جملة رقم {i} {d}", d) for d in data.DIALECTS for i in range(40)]
    fake_raw(monkeypatch, rows)
    train, val, test = data.split(data.load_clean())
    assert (len(train), len(val), len(test)) == (160, 20, 20)
    assert not set(train["clean"]) & set(test["clean"])
    assert test["label"].value_counts().nunique() == 1  # stratified: same count per dialect
