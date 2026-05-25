"""
Financial News Sentiment as a Return Predictor
Comparative Study: VADER vs TF-IDF Logistic Regression vs Lexicon Baseline
Rohini Mallikarjunaiah | Stevens Institute of Technology
Extension of: Agentic AI for Real-Time Financial Intelligence
"""

import json
import warnings
import random
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ── 1. SYNTHETIC FINANCIAL HEADLINE DATASET ──────────────────────────────────
# Realistic financial headlines with known sentiment → next-day return direction
# In production: replace with your live DuckDuckGo + YFinance pipeline data

HEADLINES = [
    # Positive headlines → stock goes UP (label=1)
    ("Apple beats earnings estimates, raises annual guidance", 1),
    ("Microsoft cloud revenue surges 28% year-over-year", 1),
    ("Fed signals pause in rate hikes, markets rally", 1),
    ("NVIDIA reports record GPU sales driven by AI demand", 1),
    ("Google announces $70B share buyback program", 1),
    ("Strong jobs report signals resilient economy", 1),
    ("Amazon AWS profit doubles, stock jumps 8%", 1),
    ("Tesla deliveries exceed analyst expectations by 12%", 1),
    ("FDA approves Pfizer breakthrough cancer therapy", 1),
    ("Meta reports first revenue growth in three quarters", 1),
    ("JPMorgan raises price targets across tech sector", 1),
    ("Oil prices fall sharply, boosting consumer spending outlook", 1),
    ("Inflation data comes in below expectations for third month", 1),
    ("Goldman Sachs upgrades S&P 500 year-end target", 1),
    ("Chipmaker TSMC expands US production capacity", 1),
    ("Salesforce beats on both earnings and revenue guidance", 1),
    ("Retail sales rise 0.8%, beating consensus estimate", 1),
    ("Berkshire Hathaway increases stake in top holdings", 1),
    ("Consumer confidence index hits 18-month high", 1),
    ("Palantir wins $480M US Army contract extension", 1),
    ("Netflix subscriber growth beats estimates by 2 million", 1),
    ("Semiconductor stocks surge on strong demand forecast", 1),
    ("Bank of America reports better-than-expected loan growth", 1),
    ("Supply chain disruptions ease, manufacturing rebounds", 1),
    ("Analyst upgrades: Goldman lifts NVDA target to $1,000", 1),
    ("Elon Musk: Tesla Cybertruck demand exceeds production capacity", 1),
    ("S&P 500 hits all-time high as tech leads rally", 1),
    ("Positive earnings surprise lifts healthcare sector broadly", 1),
    ("Strong housing starts data signals construction recovery", 1),
    ("OpenAI partnership drives Microsoft stock to record high", 1),

    # Negative headlines → stock goes DOWN (label=0)
    ("Silicon Valley Bank collapses in largest US bank failure since 2008", 0),
    ("Apple misses iPhone sales estimates amid China slowdown", 0),
    ("Fed raises rates again, warns of prolonged restrictive policy", 0),
    ("Amazon to lay off 18,000 workers in largest cuts in company history", 0),
    ("Recession fears mount as yield curve inverts further", 0),
    ("Google ad revenue disappoints, shares drop 7%", 0),
    ("Regulators launch antitrust investigation into Meta", 0),
    ("Oil prices spike on Middle East supply disruption fears", 0),
    ("Inflation unexpectedly rises, dashing hopes for rate cuts", 0),
    ("Tesla misses delivery targets, cuts prices again", 0),
    ("Microsoft warns of slowing Azure growth in Q4", 0),
    ("Weak manufacturing data signals possible economic contraction", 0),
    ("Credit Suisse collapses, triggers European banking contagion fears", 0),
    ("Retail giant files for bankruptcy amid shifting consumer trends", 0),
    ("China GDP growth misses forecast, weighs on global markets", 0),
    ("SEC charges crypto exchange with securities fraud", 0),
    ("Moody's downgrades US credit rating outlook to negative", 0),
    ("Boeing reports widening losses on 737 MAX production issues", 0),
    ("Consumer spending drops sharply as savings rate hits record low", 0),
    ("Pfizer cuts revenue forecast as COVID product demand collapses", 0),
    ("Auto industry faces headwinds as EV demand growth slows", 0),
    ("Snap revenue misses estimates, digital ad market weakens", 0),
    ("US housing market activity falls to 13-year low", 0),
    ("Goldman Sachs cuts 3,200 jobs in major restructuring", 0),
    ("Rising defaults signal stress in commercial real estate sector", 0),
    ("Disney streaming losses widen, subscriber growth stalls", 0),
    ("Geopolitical tensions escalate, risk assets sell off broadly", 0),
    ("Bank earnings disappoint as net interest margin compresses", 0),
    ("Weak jobs report raises stagflation concerns", 0),
    ("Alibaba revenue growth slows to lowest rate in company history", 0),
]

# Shuffle and split
random.shuffle(HEADLINES)
texts  = [h[0] for h in HEADLINES]
labels = [h[1] for h in HEADLINES]

split  = int(0.7 * len(HEADLINES))
X_train, X_test   = texts[:split],  texts[split:]
y_train, y_test   = labels[:split], labels[split:]

print("=" * 65)
print("  Financial News Sentiment → Return Direction Classifier")
print("  Rohini Mallikarjunaiah | Stevens Institute of Technology")
print("=" * 65)
print(f"\nDataset: {len(HEADLINES)} labeled financial headlines")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Class balance — Positive: {sum(labels)} | Negative: {len(labels)-sum(labels)}\n")


# ── 2. MODEL 1: LEXICON BASELINE (random majority) ───────────────────────────
from collections import Counter

majority_class = Counter(y_train).most_common(1)[0][0]
baseline_preds = [majority_class] * len(y_test)

def evaluate(y_true, y_pred, name):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = 0.5
    cm   = confusion_matrix(y_true, y_pred)
    tp   = cm[1,1] if cm.shape == (2,2) else 0
    fp   = cm[0,1] if cm.shape == (2,2) else 0
    fn   = cm[1,0] if cm.shape == (2,2) else 0
    tn   = cm[0,0] if cm.shape == (2,2) else 0
    return {"model": name, "accuracy": acc, "f1": f1, "auc_roc": auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

results = []
r0 = evaluate(y_test, baseline_preds, "Majority Baseline")
results.append(r0)
print(f"[1/3] Majority Baseline         Acc={r0['accuracy']:.2f}  F1={r0['f1']:.2f}  AUC={r0['auc_roc']:.2f}")


# ── 3. MODEL 2: VADER (rule-based sentiment) ─────────────────────────────────
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def vader_predict(texts):
    preds = []
    for t in texts:
        score = sia.polarity_scores(t)["compound"]
        preds.append(1 if score >= 0.0 else 0)
    return preds

vader_preds = vader_predict(X_test)
r1 = evaluate(y_test, vader_preds, "VADER (rule-based)")
results.append(r1)
print(f"[2/3] VADER Rule-Based          Acc={r1['accuracy']:.2f}  F1={r1['f1']:.2f}  AUC={r1['auc_roc']:.2f}")


# ── 4. MODEL 3: TF-IDF + LOGISTIC REGRESSION ─────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
r2 = evaluate(y_test, lr_preds, "TF-IDF + Logistic Regression")
results.append(r2)
print(f"[3/3] TF-IDF + Logistic Reg.    Acc={r2['accuracy']:.2f}  F1={r2['f1']:.2f}  AUC={r2['auc_roc']:.2f}")


# ── 5. RESULTS TABLE ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
print(f"{'Model':<35} {'Accuracy':>9} {'F1 Score':>9} {'AUC-ROC':>9}")
print("-" * 65)
for r in results:
    marker = " ◀ best" if r["model"] == max(results, key=lambda x: x["f1"])["model"] else ""
    print(f"{r['model']:<35} {r['accuracy']:>9.3f} {r['f1']:>9.3f} {r['auc_roc']:>9.3f}{marker}")
print("=" * 65)

best = max(results, key=lambda x: x["f1"])
print(f"\nBest model: {best['model']}")
print(f"  Accuracy : {best['accuracy']:.1%}")
print(f"  F1 Score : {best['f1']:.3f}")
print(f"  AUC-ROC  : {best['auc_roc']:.3f}")
print(f"  TP={best['tp']}  FP={best['fp']}  FN={best['fn']}  TN={best['tn']}")


# ── 6. FEATURE ANALYSIS (top predictive terms) ───────────────────────────────
print("\n── Top predictive terms (TF-IDF model) ──")
tfidf    = lr_pipeline.named_steps["tfidf"]
clf      = lr_pipeline.named_steps["clf"]
features = tfidf.get_feature_names_out()
coefs    = clf.coef_[0]

top_pos_idx = np.argsort(coefs)[-8:][::-1]
top_neg_idx = np.argsort(coefs)[:8]

print("\n  Bullish signals (→ predicts UP):")
for i in top_pos_idx:
    print(f"    '{features[i]}'  (weight: {coefs[i]:+.3f})")

print("\n  Bearish signals (→ predicts DOWN):")
for i in top_neg_idx:
    print(f"    '{features[i]}'  (weight: {coefs[i]:+.3f})")


# ── 7. ERROR ANALYSIS ────────────────────────────────────────────────────────
print("\n── Error analysis (misclassified headlines) ──")
misclassified = []
for text, true, pred in zip(X_test, y_test, lr_preds):
    if true != pred:
        misclassified.append({
            "headline": text,
            "true_label": "UP" if true == 1 else "DOWN",
            "predicted":  "UP" if pred == 1 else "DOWN"
        })

for i, m in enumerate(misclassified[:4]):
    print(f"\n  [{i+1}] Headline : {m['headline'][:65]}...")
    print(f"       True     : {m['true_label']}")
    print(f"       Predicted: {m['predicted']}")


# ── 8. SAVE RESULTS JSON ──────────────────────────────────────────────────────
output = {
    "experiment": "Financial News Sentiment → Return Direction Classification",
    "author": "Rohini Mallikarjunaiah",
    "institution": "Stevens Institute of Technology",
    "date": datetime.now().strftime("%Y-%m-%d"),
    "dataset": {
        "total_samples": len(HEADLINES),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "positive_class": sum(labels),
        "negative_class": len(labels) - sum(labels)
    },
    "models_evaluated": [
        {"name": r["model"], "accuracy": round(r["accuracy"], 4),
         "f1_score": round(r["f1"], 4), "auc_roc": round(r["auc_roc"], 4)}
        for r in results
    ],
    "best_model": best["model"],
    "top_bullish_terms": [features[i] for i in top_pos_idx],
    "top_bearish_terms": [features[i] for i in top_neg_idx],
    "misclassified_count": len(misclassified),
    "error_rate": round(len(misclassified) / len(X_test), 4)
}

with open("results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n\nResults saved to results.json")
print("\nDone. Add this file to your Agentic-Stock-Insights GitHub repo.")
