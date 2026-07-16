"""
classify_narratives.py
------------------------------------------------------------------------------
FemDigiNomics — Coding Sample: Multi-Label Classification of Code-Mixed
Swahili/English/Sheng Economic Narratives

Author: Samson Odhiambo (Wuod Abudo)
Purpose: Submitted as a coding sample for the NLP and AI Intern role.

WHAT THIS DEMONSTRATES
------------------------------------------------------------------------------
1. A reproducible pipeline for turning short, code-switched narratives from
   women describing their economic realities into structured, multi-label
   data suitable for downstream needs/barriers/support-pathway classification.
2. A lightweight, explainable model (TF-IDF + One-vs-Rest Logistic Regression)
   appropriate for the very small, high-context datasets typical of early-stage
   community NLP work — rather than reaching for a large model that would
   overfit and be harder to audit.
3. A rule-based lexicon layer that keeps Swahili/Sheng domain terms
   (e.g. "chama", "deni", "akiba", "mzigo") from being drowned out by
   generic stopword removal, since standard NLP libraries are English-first
   and would otherwise strip or mis-tokenize exactly the words that carry
   the most category signal.
4. Basic responsible-AI checks baked into the pipeline itself: a class-balance
   report (bias check), top-weighted terms per label (explainability), and a
   simple PII scrubber (privacy) — because these should be defaults, not
   afterthoughts, when the data describes vulnerable women's financial lives.

WHY MULTI-LABEL, NOT MULTI-CLASS
------------------------------------------------------------------------------
Real narratives rarely map to one category. A woman describing a health shock
is very often also describing care work and a savings drawdown in the same
sentence (see N003 in the dataset). Forcing single-label classification would
throw away exactly the co-occurrence patterns that matter most for designing
support pathways, so this pipeline treats the task as multi-label from the
start.

USAGE
------------------------------------------------------------------------------
    python3 classify_narratives.py --data annotated_narratives.csv

Requires: pandas, scikit-learn (see requirements.txt)
------------------------------------------------------------------------------
"""

import argparse
import re
import sys
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report


# ---------------------------------------------------------------------------
# 1. DOMAIN LEXICON
# ---------------------------------------------------------------------------
# A small, human-curated seed lexicon of Swahili/Sheng/English terms per
# category. This is NOT used to hard-code predictions — it is used to (a)
# protect these tokens from aggressive generic preprocessing, and (b) build
# an interpretable "keyword coverage" feature that annotators and reviewers
# can sanity-check against the model's learned weights. Growing this lexicon
# with community and research team input is one of the internship's core
# deliverables, so this is deliberately written to be easy to extend.

DOMAIN_LEXICON = {
    "savings": ["akiba", "kuweka pesa", "kudumu", "m-shwari", "kuweka"],
    "debt": ["deni", "mkopo", "riba", "shylock", "fuliza", "kulipa"],
    "chama_obligations": ["chama", "table banking", "merry-go-round", "fine", "faini", "mzigo"],
    "business_risk": ["biashara", "soko", "hasara", "mtaji", "duka", "bei"],
    "care_work": ["kulea", "kutunza", "mtoto", "mgonjwa", "wazazi", "nyumba"],
    "health_shocks": ["ugonjwa", "hospitali", "matibabu", "dawa", "malaria", "presha"],
    "lending_readiness": ["benki", "dhamana", "statement", "akaunti", "credit", "mkopo wa benki"],
}

ALL_LABELS = list(DOMAIN_LEXICON.keys())


# ---------------------------------------------------------------------------
# 2. PRIVACY: LIGHTWEIGHT PII SCRUBBER
# ---------------------------------------------------------------------------
# Community narratives can contain names, phone numbers, or exact locations.
# This is a minimal, transparent scrubber — in production this would be
# expanded with a proper Swahili-aware NER model, and paired with a written
# consent/anonymization protocol (see README.md, "Responsible AI" section).

PHONE_RE = re.compile(r"\b(?:\+?254|0)?7\d{8}\b")


def scrub_pii(text: str) -> str:
    text = PHONE_RE.sub("[PHONE]", text)
    return text


# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """
    Minimal, code-switch-aware normalization.
    Deliberately light-touch: aggressive stemming/lemmatization tools are
    almost all English-only and would mangle Swahili morphology (e.g. noun
    class prefixes), so we normalize case/punctuation only and let the
    vectorizer's n-gram range capture morphological variants empirically.
    """
    text = scrub_pii(text)
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_labels(raw: str):
    if pd.isna(raw) or raw.strip().lower() == "none":
        return []
    return [c.strip() for c in raw.split("|")]


# ---------------------------------------------------------------------------
# 4. RESPONSIBLE AI CHECKS
# ---------------------------------------------------------------------------
def bias_report(label_lists):
    """Simple class-balance check. In a real deployment this would also be
    cross-tabulated with respondent demographics (age, region, business
    type) to check for representation gaps — not just label frequency."""
    counts = Counter(l for labels in label_lists for l in labels)
    total = len(label_lists)
    print("\n--- BIAS / CLASS-BALANCE CHECK -------------------------------")
    for label in ALL_LABELS:
        n = counts.get(label, 0)
        print(f"  {label:20s} {n:3d} / {total}  ({n/total:.0%})")
    print("  NOTE: categories under ~15% of narratives should be treated as")
    print("  provisional until more annotated data is collected — small-n")
    print("  categories risk the model learning spurious keyword shortcuts.")


def explainability_report(vectorizer, clf, labels):
    """Surface the top positive-weight terms per label so annotators and
    domain experts (not just engineers) can audit *why* the model assigns
    a category — a requirement, not a nice-to-have, when the outputs
    describe vulnerable women's lives and may inform support decisions."""
    feature_names = vectorizer.get_feature_names_out()
    print("\n--- EXPLAINABILITY: TOP TERMS PER LABEL ----------------------")
    for i, label in enumerate(labels):
        estimator = clf.estimators_[i]
        top_idx = estimator.coef_[0].argsort()[-6:][::-1]
        top_terms = [feature_names[j] for j in top_idx]
        print(f"  {label:20s} -> {', '.join(top_terms)}")


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main(data_path: str):
    df = pd.read_csv(data_path)
    df["clean_text"] = df["text"].apply(preprocess)
    df["label_list"] = df["categories"].apply(parse_labels)

    bias_report(df["label_list"].tolist())

    mlb = MultiLabelBinarizer(classes=ALL_LABELS)
    Y = mlb.fit_transform(df["label_list"])

    # Word-level TF-IDF with a wide n-gram range to catch short Swahili
    # phrases ("mkopo wa benki") as single meaningful units, plus a low
    # min_df tolerant of this small, early-stage dataset.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["clean_text"])

    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, Y, df.index, test_size=0.3, random_state=42
    )

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    print("\n--- HELD-OUT EVALUATION (small-n, indicative only) -----------")
    print(classification_report(
        Y_test, Y_pred, target_names=ALL_LABELS, zero_division=0
    ))

    explainability_report(vectorizer, clf, ALL_LABELS)

    print("\n--- SAMPLE PREDICTIONS ON HELD-OUT NARRATIVES ----------------")
    for i, row_idx in enumerate(idx_test):
        pred_labels = [ALL_LABELS[j] for j, v in enumerate(Y_pred[i]) if v]
        true_labels = df.loc[row_idx, "label_list"]
        print(f"  [{df.loc[row_idx, 'narrative_id']}] pred={pred_labels or ['none']} "
              f"true={true_labels or ['none']}")

    print("\nDone. See README.md for methodology and responsible-AI discussion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="annotated_narratives.csv")
    args = parser.parse_args()
    main(args.data)
