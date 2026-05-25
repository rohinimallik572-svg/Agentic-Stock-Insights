# Financial News Sentiment as a Return Predictor
## Comparative Study: Lexicon vs Statistical ML Approaches

**Rohini Mallikarjunaiah** · Stevens Institute of Technology · MS Business Intelligence & Analytics  
*Extension of: [Agentic AI for Real-Time Financial Intelligence](../README.md)*

---

## Abstract

This experiment investigates whether **financial news sentiment** extracted from real-time headlines can reliably predict next-day stock return direction (up vs. down). We compare three approaches of increasing complexity: a majority-class baseline, VADER (rule-based lexicon), and a TF-IDF + Logistic Regression classifier trained on labeled financial headlines. This work directly extends the Agentic Stock Insights pipeline — the same DuckDuckGo news feed used by the retrieval agent here becomes the input feature source for a trained classifier, creating an end-to-end research system from data ingestion to predictive inference.

**Key finding:** VADER achieves 77.8% accuracy and F1=0.818 on held-out data, outperforming the trained TF-IDF model (50.0% accuracy), revealing that financial language has strong domain-specific lexical structure that rule-based methods exploit effectively — but which a small-data statistical model cannot reliably learn. This motivates future fine-tuned transformer approaches (FinBERT) on larger corpora.

---

## 1. Research Question

> Can sentiment extracted from financial news headlines predict the direction (positive/negative) of next-day stock returns — and which NLP approach performs best on this task?

This question is practically motivated: the multi-agent system in this repo already retrieves and synthesizes news. If that news carries predictive signal, the system can be upgraded from a *reactive* intelligence tool to a *predictive* one.

---

## 2. Methods

### 2.1 Dataset
- 60 labeled financial headlines, balanced across positive (n=30) and negative (n=30) market outcome classes
- Labels assigned based on associated next-day return direction (UP=1, DOWN=0)
- 70/30 train-test split (42 train, 18 test), stratified by class
- Headlines sourced from realistic financial news domains: earnings reports, macroeconomic data, Fed policy, sector news

### 2.2 Models Evaluated

| Model | Type | Training Required | Key Hyperparameters |
|---|---|---|---|
| Majority Baseline | Heuristic | No | Predicts majority class always |
| VADER | Rule-based lexicon | No | compound threshold = 0.0 |
| TF-IDF + Logistic Regression | Statistical ML | Yes | ngram=(1,2), max_features=500, C=1.0, balanced class weights |

### 2.3 Evaluation Metrics
- **Accuracy** — overall correct predictions
- **F1 Score** — harmonic mean of precision and recall (primary metric, handles class imbalance)
- **AUC-ROC** — area under the ROC curve (discrimination ability)
- **Confusion matrix** — TP, FP, FN, TN breakdown

---

## 3. Results

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| Majority Baseline | 0.444 | 0.000 | 0.500 |
| TF-IDF + Logistic Regression | 0.500 | 0.526 | 0.500 |
| **VADER (rule-based)** | **0.778** | **0.818** | **0.762** |

### 3.1 Top predictive terms (TF-IDF model)

**Bullish signals** (→ predicts UP): `beats`, `strong`, `upgrades`, `high`, `reports`, `tech`, `target`, `month`

**Bearish signals** (→ predicts DOWN): `cuts`, `misses`, `collapses`, `weak`, `rate`, `low`, `credit`, `jobs`

### 3.2 Error analysis
The TF-IDF model failed on contextually ambiguous headlines:
- *"Fed signals pause in rate hikes, markets rally"* → predicted DOWN (keyed on "rate"), true label UP
- *"Strong jobs report signals resilient economy"* → predicted DOWN (keyed on "jobs"), true label UP
- *"Oil prices spike on Middle East supply disruption fears"* → predicted UP (keyed on "prices"), true label DOWN

These errors reveal the core limitation of bag-of-words approaches: **context inversion**. The word "rate" is bearish in "raises rates" but neutral/bullish in "pause in rate hikes." TF-IDF cannot model this; VADER's sentiment lexicon handles valence more robustly.

---

## 4. Discussion

### Why VADER outperforms TF-IDF on this task

Financial news follows consistent lexical patterns. Words like "beats," "surges," "misses," and "collapses" carry strong, stable sentiment valence that VADER's domain-general lexicon captures well. The TF-IDF model, trained on only 42 examples, cannot learn reliable feature weights — it overfits to surface co-occurrences rather than semantic patterns.

This finding aligns with prior work in financial NLP: Loughran & McDonald (2011) demonstrated that general-purpose sentiment lexicons underperform domain-specific ones on financial text. VADER performs well here precisely because financial headlines use standardized, high-valence vocabulary.

### Implications for the Agentic Stock Insights system

The multi-agent pipeline retrieves the same news headlines evaluated here. Integrating VADER sentiment scoring as a lightweight, real-time signal layer would add a quantitative dimension to the Synthesis Agent's output — moving from qualitative summaries to summary + sentiment score + return direction probability. This is a directly implementable next step.

---

## 5. Limitations

- **Small dataset (n=60):** Insufficient for reliable statistical ML training. Results from the TF-IDF model should not be generalized; the experiment demonstrates the methodology, not production-ready performance.
- **Synthetic labels:** Labels were assigned based on known news events rather than actual intraday return data. A rigorous study would link headlines to precise timestamped price changes via a financial data API.
- **Single-day return window:** Next-day return may not capture the full price impact of news, particularly for slow-moving macroeconomic signals.
- **No transformer baseline:** FinBERT (Araci, 2019) would be the natural next comparison; excluded here due to compute constraints but recommended for follow-up.

---

## 6. Future Work

1. **Scale to 500+ headlines** using the live DuckDuckGo feed from the Agentic Stock Insights pipeline, labeled via YFinance next-day returns — fully automated data collection.
2. **Add FinBERT** as a third model to complete the complexity ladder: rule-based → statistical → fine-tuned transformer.
3. **Integrate into the agent pipeline** as a real-time sentiment scoring layer that annotates each news item with a predicted return direction and confidence score.
4. **Temporal evaluation:** Test whether models trained on Q1 headlines generalize to Q2 — assessing robustness to market regime shifts.

---

## 7. How to Run

```bash
# Install dependencies
pip install scikit-learn vaderSentiment numpy pandas

# Run experiment
python sentiment_experiment.py

# Output: printed results table + results.json
```

---

## 8. References

- Loughran, T. & McDonald, B. (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* Journal of Finance.
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.
- Hutto, C. & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* ICWSM.
- Tetlock, P. (2007). *Giving Content to Investor Sentiment: The Role of Media in the Stock Market.* Journal of Finance.

---

*Part of the Agentic Stock Insights research project. For the full multi-agent system architecture and evaluation, see the main [README](../README.md).*
