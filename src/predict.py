"""Guess the dialect of Arabic sentences with the trained model.

    python src/predict.py                       # a few example sentences
    python src/predict.py "ازيك عامل ايه"        # your own sentence
"""

import sys
from pathlib import Path

import joblib

from preprocess import normalize_arabic

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "dialect_model.joblib"

EXAMPLES = [
    "انا مش فاهم انت عايز ايه بالظبط",
    "شو عم تعمل هلق؟ بدك نطلع سوا",
    "شن تبي توا؟ نبي نمشي للسوق",
    "واش نتا بخير؟ بغيت نشوفك غدا",
    "انت داير شنو؟ زول ما عندو موضوع",
]


def main():
    model = joblib.load(MODEL_PATH)
    texts = sys.argv[1:] or EXAMPLES
    for text, label in zip(texts, model.predict([normalize_arabic(t) for t in texts])):
        print(f"{label:<10}| {text}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
