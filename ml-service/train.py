"""
RouteDesk - Step 2: Dataset Preparation & ML Baseline
======================================================
Dataset : Tobi-Bueck/customer-support-tickets  (Hugging Face)
Task    : Multi-class ticket classification (queue prediction)
Model   : TF-IDF + LogisticRegression (scikit-learn pipeline)

Run:
    python train.py

Artifacts saved to ml-service/artifacts/
"""

from __future__ import annotations

import io
import json
import platform
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# Force UTF-8 output so any Unicode in dataset prints correctly on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED   = 42
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DATASET_NAME  = "Tobi-Bueck/customer-support-tickets"
DATASET_SPLIT = "train"   # dataset exposes only one split; we create our own

SEP = "=" * 70

def banner(msg: str) -> None:
    print(f"\n{SEP}\n{msg}\n{SEP}")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for duplicate-group detection."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# ===========================================================================
# 1. Environment
# ===========================================================================

banner("1. ENVIRONMENT")
print(f"Python      : {sys.version}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"NumPy       : {np.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"joblib      : {joblib.__version__}")
print(f"Platform    : {platform.platform()}")

# ===========================================================================
# 2. Load dataset
# ===========================================================================

banner("2. LOADING DATASET")
print(f"Repo: {DATASET_NAME}")

ds = load_dataset(DATASET_NAME)
print(f"Splits available: {list(ds.keys())}")

try:
    ds_info         = ds[DATASET_SPLIT].info
    dataset_revision = str(getattr(ds_info, "version", "unknown"))
    print(f"Dataset name : {ds_info.dataset_name}")
    print(f"Dataset ver  : {dataset_revision}  (HF version field, not a Git commit hash)")
except Exception as exc:
    print(f"[WARN] Could not read dataset version: {exc}")
    dataset_revision = "unknown"

raw = ds[DATASET_SPLIT].to_pandas()
print(f"\nRaw shape   : {raw.shape}")
print(f"Columns     : {list(raw.columns)}")
print(f"\nDtypes:\n{raw.dtypes}")

# ===========================================================================
# 3. Inspect real column values
# ===========================================================================

banner("3. COLUMN INSPECTION")

lang_cols  = [c for c in raw.columns if "lang"    in c.lower() or "language" in c.lower()]
queue_cols = [c for c in raw.columns if "queue"   in c.lower() or "category" in c.lower() or "label" in c.lower()]
text_cols  = [c for c in raw.columns if "subject" in c.lower() or "body"     in c.lower()
              or "text" in c.lower() or "message" in c.lower() or "content"  in c.lower()]

print(f"Likely language columns : {lang_cols}")
print(f"Likely queue columns    : {queue_cols}")
print(f"Likely text columns     : {text_cols}")

for col in lang_cols:
    print(f"\n'{col}' value counts (top 20):")
    print(raw[col].value_counts().head(20).to_string())

for col in queue_cols:
    print(f"\n'{col}' value counts:")
    print(raw[col].value_counts().to_string())

for col in text_cols:
    null_cnt  = int(raw[col].isna().sum())
    empty_cnt = int((raw[col].astype(str) == "").sum())
    print(f"\n'{col}' -- null: {null_cnt}, empty string: {empty_cnt}")

# Select working columns
LANG_COL  = lang_cols[0]  if lang_cols  else None
QUEUE_COL = queue_cols[0] if queue_cols else None
SUBJ_COL  = next((c for c in raw.columns if "subject" in c.lower()), None)
BODY_COL  = next((c for c in raw.columns if "body"    in c.lower()), None)

print(f"\nSelected -> lang: '{LANG_COL}', queue: '{QUEUE_COL}', subject: '{SUBJ_COL}', body: '{BODY_COL}'")

# ===========================================================================
# 4. Data preparation
# ===========================================================================

banner("4. DATA PREPARATION")

df           = raw.copy()
initial_rows = len(df)
exclusions: list[dict] = []

# --- 4a. Filter English -------------------------------------------------------
# Language column uses ISO codes: 'en' = English, 'de' = German
if LANG_COL:
    before  = len(df)
    df      = df[df[LANG_COL].str.lower().str.strip() == "en"].copy()
    dropped = before - len(df)
    exclusions.append({"reason": "non-English language (lang != 'en')", "rows_dropped": dropped})
    print(f"[FILTER] Non-English : {dropped} dropped -> {len(df)} remain")
else:
    print("[WARN] No language column -- skipping language filter.")

# --- 4b. Combine subject + body -----------------------------------------------
def combine_text(row: pd.Series) -> str:
    parts = []
    if SUBJ_COL and pd.notna(row.get(SUBJ_COL)) and str(row[SUBJ_COL]).strip():
        parts.append(str(row[SUBJ_COL]).strip())
    if BODY_COL and pd.notna(row.get(BODY_COL)) and str(row[BODY_COL]).strip():
        parts.append(str(row[BODY_COL]).strip())
    return " ".join(parts)

df["combined_text"] = df.apply(combine_text, axis=1)

before  = len(df)
df      = df[df["combined_text"].str.strip() != ""].copy()
dropped = before - len(df)
exclusions.append({"reason": "empty combined text", "rows_dropped": dropped})
print(f"[FILTER] Empty text  : {dropped} dropped -> {len(df)} remain")

# --- 4c. Drop missing queue ---------------------------------------------------
if QUEUE_COL:
    before  = len(df)
    df      = df[df[QUEUE_COL].notna()].copy()
    df      = df[df[QUEUE_COL].astype(str).str.strip() != ""].copy()
    dropped = before - len(df)
    exclusions.append({"reason": "missing/empty queue label", "rows_dropped": dropped})
    print(f"[FILTER] No label   : {dropped} dropped -> {len(df)} remain")
    df["label"] = df[QUEUE_COL].astype(str).str.strip()
else:
    raise RuntimeError("No queue column detected -- cannot proceed without a target label.")

# --- 4d. Normalize for duplicate detection ------------------------------------
df["norm_text"] = df["combined_text"].apply(normalize_text)

# ===========================================================================
# 5. Duplicate analysis
# ===========================================================================

banner("5. DUPLICATE ANALYSIS")

group_label_sets = df.groupby("norm_text")["label"].apply(set)
conflicting      = group_label_sets[group_label_sets.apply(len) > 1]
consistent       = group_label_sets[group_label_sets.apply(len) == 1]

print(f"Unique normalized-text groups : {len(group_label_sets)}")
print(f"  Consistent (one label)      : {len(consistent)}")
print(f"  Conflicting (multi-label)   : {len(conflicting)}")

if len(conflicting) > 0:
    print("\nConflicting groups (first 10 shown):")
    for norm, labels in list(conflicting.items())[:10]:
        count = int((df["norm_text"] == norm).sum())
        print(f"  [{count} rows] labels={set(labels)} | text='{norm[:80]}'")
    conflicting_norms = set(conflicting.index)
    before  = len(df)
    df      = df[~df["norm_text"].isin(conflicting_norms)].copy()
    dropped = before - len(df)
    exclusions.append({
        "reason": f"ambiguous duplicates ({len(conflicting_norms)} groups with conflicting labels)",
        "rows_dropped": dropped,
    })
    print(f"\n[FILTER] Ambiguous dups : {dropped} dropped -> {len(df)} remain")
else:
    print("No conflicting groups found.")

print(f"\nRows retained: {len(df)} / {initial_rows} initial")
label_counts = df["label"].value_counts()
print(f"\nQueue distribution:\n{label_counts.to_string()}")

# --- Remove classes with < 3 samples (cannot form 3-way splits) --------------
MIN_SPLIT = 3
small_cls = label_counts[label_counts < MIN_SPLIT]
if len(small_cls):
    print(f"\n[WARN] Classes with < {MIN_SPLIT} samples (excluded -- cannot split reliably):")
    print(small_cls.to_string())
    before  = len(df)
    df      = df[~df["label"].isin(small_cls.index)].copy()
    dropped = before - len(df)
    exclusions.append({
        "reason": f"< {MIN_SPLIT} samples, cannot split: {list(small_cls.index)}",
        "rows_dropped": dropped,
    })
    print(f"[FILTER] Tiny classes: {dropped} dropped -> {len(df)} remain")
    label_counts = df["label"].value_counts()

classes_retained = sorted(df["label"].unique().tolist())
print(f"\nClasses retained ({len(classes_retained)}): {classes_retained}")

# ===========================================================================
# 6. Leakage-safe group-aware splits
# ===========================================================================

banner("6. LEAKAGE-SAFE DATA SPLITS")

rng             = np.random.default_rng(RANDOM_SEED)
group_to_label  = df.groupby("norm_text")["label"].first()
all_grps        = group_to_label.index.tolist()
all_lbls        = group_to_label.values.tolist()
shuf_idx        = rng.permutation(len(all_grps))
all_grps        = [all_grps[i] for i in shuf_idx]
all_lbls        = [all_lbls[i] for i in shuf_idx]
groups_df       = pd.DataFrame({"norm_text": all_grps, "label": all_lbls})

# Report unique groups per class
grp_label_counts = Counter(all_lbls)
print("Unique text groups per class:")
for cls, cnt in sorted(grp_label_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {cnt} groups")

# Exclude classes with < 3 unique groups (cannot reliably split into 3 sets)
tiny_grp_classes = [c for c, n in grp_label_counts.items() if n < 3]
if tiny_grp_classes:
    print(f"\n[LIMITATION] Classes with < 3 unique text groups -- excluded: {tiny_grp_classes}")
    df = df[~df["label"].isin(tiny_grp_classes)].copy()
    exclusions.append({
        "reason": f"< 3 unique text groups for splitting: {tiny_grp_classes}",
        "rows_dropped": 0,
    })
    group_to_label = df.groupby("norm_text")["label"].first()
    all_grps       = group_to_label.index.tolist()
    all_lbls       = group_to_label.values.tolist()
    shuf_idx       = rng.permutation(len(all_grps))
    all_grps       = [all_grps[i] for i in shuf_idx]
    all_lbls       = [all_lbls[i] for i in shuf_idx]
    groups_df      = pd.DataFrame({"norm_text": all_grps, "label": all_lbls})
    classes_retained = sorted(df["label"].unique().tolist())

# Stratified group split: 70% train / 15% val / 15% test
stratified_ok = True
try:
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    tr_idx, tmp_idx = next(sss1.split(groups_df["norm_text"], groups_df["label"]))
    train_grps = set(groups_df.iloc[tr_idx]["norm_text"])
    tmp_df     = groups_df.iloc[tmp_idx].copy().reset_index(drop=True)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    v_idx, t_idx = next(sss2.split(tmp_df["norm_text"], tmp_df["label"]))
    val_grps  = set(tmp_df.iloc[v_idx]["norm_text"])
    test_grps = set(tmp_df.iloc[t_idx]["norm_text"])
except ValueError as exc:
    print(f"[WARN] Stratified split failed ({exc}); falling back to random group split.")
    stratified_ok = False
    n      = len(all_grps)
    n_tr   = int(0.70 * n)
    n_v    = int(0.15 * n)
    train_grps = set(all_grps[:n_tr])
    val_grps   = set(all_grps[n_tr : n_tr + n_v])
    test_grps  = set(all_grps[n_tr + n_v :])

# Verify no leakage
assert train_grps.isdisjoint(val_grps),  "LEAK: train and val share text groups!"
assert train_grps.isdisjoint(test_grps), "LEAK: train and test share text groups!"
assert val_grps.isdisjoint(test_grps),   "LEAK: val and test share text groups!"
print("[OK] No text-group overlap across splits.")

train_df = df[df["norm_text"].isin(train_grps)].copy()
val_df   = df[df["norm_text"].isin(val_grps)].copy()
test_df  = df[df["norm_text"].isin(test_grps)].copy()

print(f"\nSplit sizes (stratified={stratified_ok}):")
print(f"  Train : {len(train_df):>6} rows  ({len(train_grps)} unique text groups)")
print(f"  Val   : {len(val_df):>6} rows  ({len(val_grps)} unique text groups)")
print(f"  Test  : {len(test_df):>6} rows  ({len(test_grps)} unique text groups)")
print(f"  Total : {len(train_df) + len(val_df) + len(test_df)}")

print(f"\nTrain class distribution:\n{train_df['label'].value_counts().to_string()}")
print(f"\nVal class distribution:\n{val_df['label'].value_counts().to_string()}")
print(f"\nTest class distribution:\n{test_df['label'].value_counts().to_string()}")

# Warn on classes absent from val / test
tr_cls   = set(train_df["label"].unique())
val_cls  = set(val_df["label"].unique())
test_cls = set(test_df["label"].unique())
if tr_cls - val_cls:
    print(f"\n[WARN] Classes in train but absent from val  : {tr_cls - val_cls}")
if tr_cls - test_cls:
    print(f"[WARN] Classes in train but absent from test : {tr_cls - test_cls}")

# ===========================================================================
# 7. Train baseline
# ===========================================================================

banner("7. TRAINING BASELINE")

X_train, y_train = train_df["combined_text"].tolist(), train_df["label"].tolist()
X_val,   y_val   = val_df["combined_text"].tolist(),   val_df["label"].tolist()
X_test,  y_test  = test_df["combined_text"].tolist(),  test_df["label"].tolist()


def build_pipeline(class_weight=None) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            min_df=2,
            ngram_range=(1, 2),
            max_features=50_000,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",         # lbfgs handles multinomial natively
            class_weight=class_weight,
            C=1.0,
        )),
    ])


# Variant A -- standard weights
pipe_std    = build_pipeline(None)
pipe_std.fit(X_train, y_train)
val_f1_std  = f1_score(y_val, pipe_std.predict(X_val), average="macro", zero_division=0)
print(f"Variant A (standard)   -- Val Macro F1: {val_f1_std:.4f}")

# Variant B -- balanced weights
pipe_bal    = build_pipeline("balanced")
pipe_bal.fit(X_train, y_train)
val_f1_bal  = f1_score(y_val, pipe_bal.predict(X_val), average="macro", zero_division=0)
print(f"Variant B (balanced)   -- Val Macro F1: {val_f1_bal:.4f}")

# Select better model on validation macro F1
if val_f1_bal >= val_f1_std:
    selected_pipe, selected_label, selected_val_f1 = pipe_bal, "balanced", val_f1_bal
else:
    selected_pipe, selected_label, selected_val_f1 = pipe_std, "standard", val_f1_std

print(f"\nSelected model: class_weight='{selected_label}'  (Val Macro F1 = {selected_val_f1:.4f})")

# ===========================================================================
# 8. Evaluate on held-out test set
# ===========================================================================

banner("8. TEST-SET EVALUATION (selected model)")

test_preds    = selected_pipe.predict(X_test)
test_acc      = accuracy_score(y_test, test_preds)
test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)

print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Macro F1 : {test_macro_f1:.4f}")

majority_cls = Counter(y_train).most_common(1)[0][0]
maj_preds    = [majority_cls] * len(y_test)
maj_acc      = accuracy_score(y_test, maj_preds)
maj_f1       = f1_score(y_test, maj_preds, average="macro", zero_division=0)
print(f"\nMajority-class baseline ('{majority_cls}'):")
print(f"  Accuracy : {maj_acc:.4f}")
print(f"  Macro F1 : {maj_f1:.4f}")

print(f"\nPer-class report:\n{classification_report(y_test, test_preds, zero_division=0)}")

# --- High-score investigation -------------------------------------------------
if test_macro_f1 > 0.98:
    print("\n" + "!" * 70)
    print("WARNING: Macro F1 > 0.98 -- INVESTIGATING FOR LEAKAGE")
    print("!" * 70)
    train_norms = set(train_df["norm_text"])
    test_norms  = set(test_df["norm_text"])
    overlap     = train_norms & test_norms
    print(f"Normalized text-group overlap (train & test): {len(overlap)}")
    if overlap:
        print("  POTENTIAL LEAKAGE: group split may be broken -- inspect carefully.")
    else:
        print("  No text-group overlap found. High score likely reflects dataset properties.")
    tr_pairs  = set(zip(train_df["norm_text"], train_df["label"]))
    te_pairs  = set(zip(test_df["norm_text"],  test_df["label"]))
    exact_dup = tr_pairs & te_pairs
    print(f"  Exact (text, label) pairs shared across train/test: {len(exact_dup)}")

# ===========================================================================
# 9. Confusion matrix
# ===========================================================================

banner("9. CONFUSION MATRIX")
classes_used = sorted(set(y_test) | set(test_preds))
cm_arr       = confusion_matrix(y_test, test_preds, labels=classes_used)
cm_df        = pd.DataFrame(cm_arr, index=classes_used, columns=classes_used)
print(cm_df.to_string())

# ===========================================================================
# 10. Save artifacts
# ===========================================================================

banner("10. SAVING ARTIFACTS")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline (large binary -- excluded from Git via .gitignore)
model_path = ARTIFACTS_DIR / "tfidf_lr_pipeline.joblib"
joblib.dump(selected_pipe, model_path)
print(f"[SAVED] Pipeline      -> {model_path}")

# Class list (small -- eligible for Git)
labels_path = ARTIFACTS_DIR / "classes.json"
labels_path.write_text(json.dumps(classes_retained, indent=2), encoding="utf-8")
print(f"[SAVED] Classes       -> {labels_path}")

# Confusion matrix CSV (small -- eligible for Git)
cm_path = ARTIFACTS_DIR / "confusion_matrix.csv"
cm_df.to_csv(cm_path)
print(f"[SAVED] Confusion CSV -> {cm_path}")

# Per-class metrics dict
cr_dict = classification_report(y_test, test_preds, zero_division=0, output_dict=True)

# Metrics report (small -- eligible for Git)
metrics = {
    "dataset": {
        "name":       DATASET_NAME,
        "revision":   dataset_revision,
        "split_used": DATASET_SPLIT,
        "note":       "revision is the HF dataset version field (0.0.0), not a Git commit hash",
    },
    "data_prep": {
        "initial_raw_rows":    initial_rows,
        "rows_after_cleaning": len(df),
        "classes_retained":    classes_retained,
        "num_classes":         len(classes_retained),
        "exclusions":          exclusions,
    },
    "splits": {
        "strategy":            "stratified group split (group=normalized text)",
        "stratified":          stratified_ok,
        "random_seed":         RANDOM_SEED,
        "train_rows":          len(train_df),
        "val_rows":            len(val_df),
        "test_rows":           len(test_df),
        "train_unique_groups": len(train_grps),
        "val_unique_groups":   len(val_grps),
        "test_unique_groups":  len(test_grps),
    },
    "model": {
        "type":         "TfidfVectorizer + LogisticRegression (sklearn Pipeline)",
        "class_weight": selected_label,
        "C":            1.0,
        "max_iter":     1000,
        "solver":       "lbfgs",
        "note":         "multinomial handled natively by lbfgs (multi_class removed in sklearn 1.7+)",
        "ngram_range":  [1, 2],
        "max_features": 50000,
        "sublinear_tf": True,
        "min_df":       2,
        "random_seed":  RANDOM_SEED,
    },
    "validation": {
        "standard_macro_f1": round(val_f1_std, 6),
        "balanced_macro_f1": round(val_f1_bal, 6),
        "selected_variant":  selected_label,
        "selected_macro_f1": round(selected_val_f1, 6),
    },
    "test_evaluation": {
        "accuracy":                   round(test_acc, 6),
        "macro_f1":                   round(test_macro_f1, 6),
        "majority_baseline_accuracy": round(maj_acc, 6),
        "majority_baseline_macro_f1": round(maj_f1, 6),
        "majority_class":             majority_cls,
        "per_class": {
            cls: {
                "precision": round(v["precision"], 6),
                "recall":    round(v["recall"],    6),
                "f1-score":  round(v["f1-score"],  6),
                "support":   int(v["support"]),
            }
            for cls, v in cr_dict.items()
            if cls not in ("accuracy", "macro avg", "weighted avg")
        },
    },
    "environment": {
        "python_version":  sys.version,
        "sklearn_version": sklearn.__version__,
        "numpy_version":   np.__version__,
        "pandas_version":  pd.__version__,
        "joblib_version":  joblib.__version__,
        "platform":        platform.platform(),
    },
    "inference_dependencies": {
        "description":    "Minimum versions required to load and run the saved pipeline",
        "scikit-learn":   f">={sklearn.__version__}",
        "joblib":         f">={joblib.__version__}",
        "numpy":          f">={np.__version__}",
    },
}

metrics_path = ARTIFACTS_DIR / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"[SAVED] Metrics JSON  -> {metrics_path}")

banner("TRAINING COMPLETE")
print(f"Artifacts dir : {ARTIFACTS_DIR.resolve()}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Macro F1 : {test_macro_f1:.4f}")
