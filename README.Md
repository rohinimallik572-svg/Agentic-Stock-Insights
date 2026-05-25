  with DDGS() as ddgs:
ERROR: Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.3-70b-versatile` in organization `org_01ksg7xgjvejfs2tgvvy4rf8fh` service tier `on_demand` on tokens per day (TPD): Limit 100000, Used 99788, Requested 720. Please try again in 7m18.912s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}

Raw results saved → ./results_raw.csv
Summary saved        → ./results_summary.csv
Markdown table saved → ./results_table.md

── Final summary ──────────────────────────────────────────
  Baseline         acc=0%  hall=100%  coh=4.04  lat=0.82s
  RAG              acc=100%  hall=0%  coh=5  lat=2.27s
  Multi-agent      acc=100%  hall=0%  coh=5  lat=3.33s

Done. Copy results_table.md content into your README.
============================================================
(base) rohinimallikarjunaiah@Mac Downloads % cat results_table.md
## 5. Findings

### 5.1 Accuracy and Hallucination

| Condition | Factual Accuracy | Hallucination Rate |
|---|---|---|
| Baseline (no retrieval) | 0% | 100% |
| RAG baseline (structured only) | 100% | 0% |
| **Multi-agent system (ours)** | **100%** | **0%** |

### 5.2 Coherence

| Condition | Mean Coherence (1–5) |
|---|---|
| Baseline (no retrieval) | 4.04 |
| RAG baseline (structured only) | 5 |
| **Multi-agent system (ours)** | **5** |

### 5.3 Latency

| Condition | Mean latency (s) | SD (s) |
|---|---|---|
| Baseline (no retrieval) | 0.82 | 0.49 |
| RAG baseline (structured only) | 2.27 | 0.71 |
| Multi-agent system (ours) | 3.33 | 2.34 |

### 5.4 Multi-agent results by ticker category

| Category | n | Factual Accuracy | Hallucination Rate | Coherence |
|---|---|---|---|---|
| Large Cap | 10 | 100% | 0% | 5 |
| Mid Cap | 10 | 100% | 0% | 5 |
| Small Cap | 7 | 100% | 0% | 5 |

*Benchmark run: 2026-05-25 · Model: llama-3.3-70b-versatile · n=30 tickers*
(base) rohinimallikarjunaiah@Mac Downloads % 
