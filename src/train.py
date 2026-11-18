"""Train and compare dialect classifiers.

1. Split 80 / 10 / 10 (stratified).
2. Baseline: always say the most common dialect (Egyptian).
3. Each model gets a small search over one hyperparameter on the validation set.
4. The best one (by validation macro F1) is retrained on train + validation and
   scored on the test set once.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from data import DIALECTS, load_clean, split

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
CLASSES = list(DIALECTS.values())


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                           max_features=500_000, dtype=np.float32)


def features(kind):
    if kind == "words":
        return word_tfidf()
    if kind == "characters":
        return char_tfidf()
    return FeatureUnion([("words", word_tfidf()), ("characters", char_tfidf())])


# name: (features, classifier factory, values to try)
MODELS = {
    "Naive Bayes (words)": ("words", lambda a: MultinomialNB(alpha=a), [0.03, 0.1, 0.3, 1.0]),
    "Logistic Regression (words)": ("words", lambda c: LogisticRegression(C=c, max_iter=3000), [1, 3, 10]),
    "Linear SVM (words)": ("words", lambda c: LinearSVC(C=c), [0.05, 0.1, 0.3, 1.0]),
    "Linear SVM (characters)": ("characters", lambda c: LinearSVC(C=c), [0.05, 0.1, 0.3, 1.0]),
    "Linear SVM (words + characters)": ("both", lambda c: LinearSVC(C=c), [0.05, 0.1, 0.3, 1.0]),
}


def score(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro")), 4)}


def plot_confusion(y, p, path):
    cm = confusion_matrix(y, p, labels=CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True dialect")
    ax.set_title("Test set, each row sums to 100%")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean(verbose=True)
    train, val, test = split(df)
    print(f"train {len(train):,}  val {len(val):,}  test {len(test):,}")
    print(train["label"].value_counts(normalize=True).round(3).to_string())

    results = {"sizes": {"train": len(train), "val": len(val), "test": len(test)},
               "label_share": train["label"].value_counts(normalize=True).round(4).to_dict(), "models": {}}

    dummy = DummyClassifier(strategy="most_frequent").fit(train["clean"], train["label"])
    p = dummy.predict(val["clean"])
    results["models"]["Always Egyptian"] = {"val": score(val["label"], p)}
    print("Always Egyptian", results["models"]["Always Egyptian"])

    cache = {}
    best = (None, None, -1)
    for name, (kind, make, values) in MODELS.items():
        if kind not in cache:
            t = time.time()
            vec = features(kind)
            cache[kind] = (vec, vec.fit_transform(train["clean"]), None)
            cache[kind] = (vec, cache[kind][1], vec.transform(val["clean"]))
            print(f"features {kind}: {cache[kind][1].shape[1]:,} ({time.time() - t:.0f}s)")
        vec, Xtr, Xva = cache[kind]
        tried = {}
        for v in values:
            t = time.time()
            clf = make(v).fit(Xtr, train["label"])
            tried[v] = score(val["label"], clf.predict(Xva))
            print(f"  {name} {v}: {tried[v]} ({time.time() - t:.0f}s)")
        v_best = max(tried, key=lambda v: tried[v]["macro_f1"])
        results["models"][name] = {"best_param": v_best, "val": tried[v_best], "tried": {str(k): s for k, s in tried.items()}}
        if tried[v_best]["macro_f1"] > best[2]:
            best = (name, v_best, tried[v_best]["macro_f1"])

    name, v, _ = best
    kind, make, _ = MODELS[name]
    print(f"best: {name} ({v}); retraining on train + val")
    full = pd.concat([train, val])
    model = make_pipeline(features(kind), make(v)).fit(full["clean"], full["label"])
    p = model.predict(test["clean"])
    results["picked"] = {"model": name, "param": v}
    results["test"] = score(test["label"], p)
    results["test"]["recall"] = {c: round(float(r), 4) for c, r in
                                 zip(CLASSES, recall_score(test["label"], p, labels=CLASSES, average=None))}
    results["test_always_egyptian"] = score(test["label"], dummy.predict(test["clean"]))
    print(classification_report(test["label"], p, digits=3))
    plot_confusion(test["label"], p, FIG_DIR / "confusion_matrix.png")
    joblib.dump(model, MODEL_DIR / "dialect_model.joblib")
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results["test"], indent=2))


if __name__ == "__main__":
    main()
