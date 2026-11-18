"""Download, clean and split the Arabic dialect tweets dataset."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import normalize_arabic

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "tweets.parquet"
DATA_URL = (
    "https://huggingface.co/datasets/amgadhasan/arabic_tweets_dialects"
    "/resolve/main/data/train-00000-of-00001.parquet"
)

DIALECTS = {"EG": "Egyptian", "LY": "Libyan", "LB": "Lebanese", "SD": "Sudanese", "MA": "Moroccan"}
RANDOM_STATE = 42
MIN_WORDS = 3  # tweets with fewer words than this after cleaning say almost nothing about the dialect


def download() -> Path:
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {RAW_PATH} ...")
        urlretrieve(DATA_URL, RAW_PATH)
    return RAW_PATH


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(download())
    df["label"] = df["dialect"].map(DIALECTS)
    return df


def load_clean(verbose: bool = False) -> pd.DataFrame:
    """Normalize the text, then drop too-short, conflicting and duplicate tweets."""
    df = load_raw()
    n0 = len(df)
    df["clean"] = df["text"].map(normalize_arabic)

    df = df[df["clean"].str.split().str.len() >= MIN_WORDS]
    n_short = n0 - len(df)

    # same tweet text labeled as two different dialects: can't be learned, drop all copies
    labels_per_text = df.groupby("clean")["label"].transform("nunique")
    conflicting = labels_per_text > 1
    n_conflict = int(conflicting.sum())
    df = df[~conflicting]

    # retweets and copies: keep one so the same text can't be in both train and test
    n_before = len(df)
    df = df.drop_duplicates(subset="clean")
    n_dupes = n_before - len(df)

    if verbose:
        print(f"Raw tweets:                   {n0:,}")
        print(f"Fewer than {MIN_WORDS} words:        -{n_short:,}")
        print(f"Same text, different dialect: -{n_conflict:,}")
        print(f"Duplicates:                   -{n_dupes:,}")
        print(f"Final:                         {len(df):,}")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    """80 / 10 / 10 train / validation / test, stratified by dialect."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=RANDOM_STATE)
    return train, val, test
