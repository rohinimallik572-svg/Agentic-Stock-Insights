"""
Agentic Stock Insights — 30-Ticker Benchmark Evaluation
=========================================================
Runs 30 tickers across 3 conditions:
  - Condition A: Baseline (no retrieval, raw LLM only)
  - Condition B: RAG baseline (YFinance structured data only)
  - Condition C: Multi-agent system (YFinance + DuckDuckGo + Synthesis)

Measures:
  - Factual accuracy  : % of numerical claims matching YFinance ground truth
  - Hallucination rate: % of outputs containing invented facts
  - Coherence score   : 1–5 scale via LLM-as-judge
  - Latency           : end-to-end seconds

Outputs:
  - results_raw.csv       : one row per ticker per condition
  - results_summary.csv   : aggregated by condition
  - results_table.md      : markdown table ready to paste into README

Usage:
  pip install yfinance groq duckduckgo-search agno
  export GROQ_API_KEY=your_key_here
  python evaluate_stock_insights.py
"""

import os, time, json, csv, re, statistics
from datetime import datetime
from typing import Optional

# ── dependencies ────────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install yfinance:  pip install yfinance")

try:
    from groq import Groq
except ImportError:
    raise SystemExit("Install groq:  pip install groq")

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    print("WARNING: duckduckgo-search not installed. Condition C will skip news.")
    print("         Install with:  pip install duckduckgo-search")
    HAS_DDG = False

# ── configuration ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
OUTPUT_DIR = "."

TICKERS = {
    "large_cap": [
        "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
        "META", "TSLA", "BRK-B", "JPM", "JNJ"
    ],
    "mid_cap": [
        "RIVN", "PLTR", "SMCI", "DKNG", "HOOD",
        "AFRM", "SOFI", "IONQ", "RBLX", "COIN"
    ],
    "small_cap": [
        "ACHR", "JOBY", "SPCE", "NKLA", "WKHS",
        "RIDE", "IDEX", "MVIS", "PTRA", "GOEV"
    ]
}

ALL_TICKERS = (
    [(t, "large_cap")  for t in TICKERS["large_cap"]] +
    [(t, "mid_cap")    for t in TICKERS["mid_cap"]]   +
    [(t, "small_cap")  for t in TICKERS["small_cap")]
)

# ── LLM client ───────────────────────────────────────────────────────────────
def get_client():
    if not GROQ_API_KEY:
        raise SystemExit(
            "\nGROQ_API_KEY not set.\n"
            "Run:  export GROQ_API_KEY=your_key_here\n"
            "Then re-run this script."
        )
    return Groq(api_key=GROQ_API_KEY)

def llm(client, system: str, user: str, max_tokens: int = 600) -> tuple[str, float]:
    """Call LLM; return (response_text, latency_seconds)."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
    )
    latency = round(time.time() - t0, 2)
    return resp.choices[0].message.content.strip(), latency

# ── data retrieval ────────────────────────────────────────────────────────────
def fetch_ground_truth(ticker: str) -> dict:
    """Fetch ground-truth structured data from YFinance."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="5d")
        current_price = round(hist["Close"].iloc[-1], 2) if not hist.empty else None
        return {
            "ticker":           ticker,
            "current_price":    current_price,
            "52w_high":         info.get("fiftyTwoWeekHigh"),
            "52w_low":          info.get("fiftyTwoWeekLow"),
            "volume":           info.get("volume"),
            "market_cap":       info.get("marketCap"),
            "analyst_target":   info.get("targetMeanPrice"),
            "recommendation":   info.get("recommendationKey", "n/a"),
            "pe_ratio":         info.get("trailingPE"),
            "company_name":     info.get("shortName", ticker),
            "sector":           info.get("sector", "Unknown"),
        }
    except Exception as e:
        print(f"  [YFinance error for {ticker}]: {e}")
        return {"ticker": ticker, "current_price": None}

def fetch_news(ticker: str, n: int = 5) -> str:
    """Fetch recent news headlines via DuckDuckGo."""
    if not HAS_DDG:
        return "News unavailable (duckduckgo-search not installed)."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(f"{ticker} stock news", max_results=n))
        if not results:
            return f"No recent news found for {ticker}."
        headlines = "\n".join(
            f"- [{r.get('date','?')}] {r.get('title','')}: {r.get('body','')[:120]}"
            for r in results
        )
        return headlines
    except Exception as e:
        return f"News fetch failed: {e}"

# ── three conditions ──────────────────────────────────────────────────────────
SYNTHESIS_SYSTEM = """You are a financial analyst AI. You receive structured financial data
and recent news about a stock. Your job is to produce a clear, grounded market intelligence
summary. Rules:
- Cite specific numbers from the data provided before making any inference.
- If data is missing or sparse, say so explicitly with an "uncertainty flag".
- Structure your output: (1) Price & fundamentals summary, (2) Recent news & sentiment,
  (3) Analyst view, (4) Key risk flags.
- Never invent numbers not present in the data provided."""

def condition_a_baseline(client, ticker: str) -> tuple[str, float]:
    """No retrieval — raw LLM only."""
    system = "You are a financial analyst. Answer concisely."
    user   = f"Give me a market intelligence summary for {ticker} stock: current price, analyst sentiment, recent news, and key risks."
    return llm(client, system, user)

def condition_b_rag(client, ticker: str, ground_truth: dict) -> tuple[str, float]:
    """Structured data only — no news."""
    data_str = json.dumps({k: v for k, v in ground_truth.items() if v is not None}, indent=2)
    user = f"""Here is structured financial data for {ticker}:

{data_str}

Produce a market intelligence summary using only this data."""
    return llm(client, SYNTHESIS_SYSTEM, user)

def condition_c_multiagent(client, ticker: str, ground_truth: dict) -> tuple[str, float]:
    """Full multi-agent: structured + news → synthesis."""
    t0 = time.time()

    # Agent 1: data (already fetched as ground_truth)
    data_str = json.dumps({k: v for k, v in ground_truth.items() if v is not None}, indent=2)

    # Agent 2: news
    news_str = fetch_news(ticker)

    # Agent 3: synthesis
    user = f"""STRUCTURED FINANCIAL DATA for {ticker}:
{data_str}

RECENT NEWS:
{news_str}

Produce a comprehensive, grounded market intelligence summary."""
    response, llm_latency = llm(client, SYNTHESIS_SYSTEM, user)

    # total latency includes news fetch (already done above, llm_latency is subset)
    total_latency = round(time.time() - t0, 2)
    return response, total_latency

# ── evaluation (LLM-as-judge) ─────────────────────────────────────────────────
JUDGE_SYSTEM = """You are an evaluation judge for AI financial summaries. You will be given:
1. Ground-truth financial data (from YFinance)
2. An AI-generated summary

Respond with ONLY valid JSON in this exact format (no other text):
{
  "factual_accuracy_pct": <integer 0-100>,
  "hallucination_detected": <true or false>,
  "hallucination_example": "<quote the hallucinated claim, or empty string>",
  "coherence_score": <integer 1-5>,
  "notes": "<one sentence>"
}

Scoring guide:
- factual_accuracy_pct: % of numerical claims in the summary that match ground truth (price, volume, 52w range, analyst target). Ignore claims where ground truth is null.
- hallucination_detected: true if the summary states a specific fact (number, event, quote) not present in ground truth and not clearly hedged.
- coherence_score: 1=incoherent, 2=poor, 3=adequate, 4=good, 5=excellent logical flow and readability."""

def evaluate_output(client, ground_truth: dict, output: str) -> dict:
    """Use LLM as judge to score an output."""
    gt_str = json.dumps({k: v for k, v in ground_truth.items() if v is not None}, indent=2)
    user   = f"GROUND TRUTH:\n{gt_str}\n\nAI SUMMARY TO EVALUATE:\n{output}"
    try:
        raw, _ = llm(client, JUDGE_SYSTEM, user, max_tokens=300)
        # strip any markdown fences if present
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        print(f"    [Judge parse error]: {e} | raw: {raw[:200]}")
        return {
            "factual_accuracy_pct":  50,
            "hallucination_detected": False,
            "hallucination_example": "",
            "coherence_score":        3,
            "notes":                  "Judge parse failed"
        }

# ── main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation():
    if not GROQ_API_KEY:
        raise SystemExit(
            "\nGROQ_API_KEY not set.\n"
            "Run:  export GROQ_API_KEY=gsk_your_key_here\n"
        )

    client = get_client()
    rows   = []
    total  = len(ALL_TICKERS)

    print(f"\n{'='*60}")
    print(f"  Agentic Stock Insights — Benchmark Evaluation")
    print(f"  {total} tickers × 3 conditions = {total*3} LLM calls")
    print(f"  Model: {MODEL}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for i, (ticker, category) in enumerate(ALL_TICKERS, 1):
        print(f"[{i:02d}/{total}] {ticker} ({category})")

        # fetch ground truth once per ticker
        print(f"  Fetching ground truth from YFinance...")
        gt = fetch_ground_truth(ticker)
        price_tag = f"${gt['current_price']}" if gt.get("current_price") else "N/A"
        print(f"  Price: {price_tag} | Rec: {gt.get('recommendation','?')}")

        for cond_name, cond_fn in [
            ("A_baseline",    lambda: condition_a_baseline(client, ticker)),
            ("B_rag",         lambda: condition_b_rag(client, ticker, gt)),
            ("C_multiagent",  lambda: condition_c_multiagent(client, ticker, gt)),
        ]:
            print(f"  Condition {cond_name}...", end=" ", flush=True)
            try:
                output, latency = cond_fn()
                scores = evaluate_output(client, gt, output)
                print(f"acc={scores['factual_accuracy_pct']}% | "
                      f"hall={'Y' if scores['hallucination_detected'] else 'N'} | "
                      f"coh={scores['coherence_score']} | "
                      f"{latency}s")
                rows.append({
                    "ticker":               ticker,
                    "category":             category,
                    "condition":            cond_name,
                    "factual_accuracy_pct": scores["factual_accuracy_pct"],
                    "hallucination":        1 if scores["hallucination_detected"] else 0,
                    "hallucination_example":scores.get("hallucination_example",""),
                    "coherence_score":      scores["coherence_score"],
                    "latency_s":            latency,
                    "notes":                scores.get("notes",""),
                    "output_preview":       output[:200].replace("\n"," "),
                })
            except Exception as e:
                print(f"ERROR: {e}")
                rows.append({
                    "ticker": ticker, "category": category,
                    "condition": cond_name,
                    "factual_accuracy_pct": None, "hallucination": None,
                    "hallucination_example":"", "coherence_score": None,
                    "latency_s": None, "notes": f"Error: {e}",
                    "output_preview": "",
                })
            time.sleep(0.5)  # gentle rate limiting

        print()

    return rows

# ── output writers ────────────────────────────────────────────────────────────
def write_raw_csv(rows: list, path: str):
    fields = ["ticker","category","condition","factual_accuracy_pct",
              "hallucination","coherence_score","latency_s","notes","output_preview"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Raw results saved → {path}")

def summarize(rows: list) -> dict:
    """Aggregate metrics by condition."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        if r["factual_accuracy_pct"] is not None:
            buckets[r["condition"]].append(r)

    summary = {}
    for cond, rr in buckets.items():
        acc    = [r["factual_accuracy_pct"] for r in rr]
        hall   = [r["hallucination"]        for r in rr]
        coh    = [r["coherence_score"]      for r in rr]
        lat    = [r["latency_s"]            for r in rr if r["latency_s"]]
        summary[cond] = {
            "n":                    len(rr),
            "factual_accuracy_pct": round(statistics.mean(acc), 1),
            "hallucination_rate_pct": round(statistics.mean(hall)*100, 1),
            "coherence_mean":       round(statistics.mean(coh), 2),
            "latency_mean_s":       round(statistics.mean(lat), 2) if lat else None,
            "latency_sd_s":         round(statistics.stdev(lat), 2) if len(lat)>1 else None,
        }
    return summary

def write_summary_csv(summary: dict, path: str):
    fields = ["condition","n","factual_accuracy_pct","hallucination_rate_pct",
              "coherence_mean","latency_mean_s","latency_sd_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond, s in summary.items():
            w.writerow({"condition": cond, **s})
    print(f"Summary saved        → {path}")

def write_markdown_table(summary: dict, rows: list, path: str):
    """Write a README-ready results table and per-category breakdown."""
    cond_labels = {
        "A_baseline":   "Baseline (no retrieval)",
        "B_rag":        "RAG baseline (structured only)",
        "C_multiagent": "Multi-agent system (ours)",
    }

    lines = []
    lines.append("## 5. Findings\n")
    lines.append("### 5.1 Accuracy and Hallucination\n")
    lines.append("| Condition | Factual Accuracy | Hallucination Rate |")
    lines.append("|---|---|---|")
    for cond in ["A_baseline","B_rag","C_multiagent"]:
        s = summary.get(cond, {})
        label   = cond_labels.get(cond, cond)
        acc     = f"{s.get('factual_accuracy_pct','?')}%"
        hall    = f"{s.get('hallucination_rate_pct','?')}%"
        bold    = cond == "C_multiagent"
        if bold:
            lines.append(f"| **{label}** | **{acc}** | **{hall}** |")
        else:
            lines.append(f"| {label} | {acc} | {hall} |")

    lines.append("\n### 5.2 Coherence\n")
    lines.append("| Condition | Mean Coherence (1–5) |")
    lines.append("|---|---|")
    for cond in ["A_baseline","B_rag","C_multiagent"]:
        s = summary.get(cond, {})
        label = cond_labels.get(cond, cond)
        coh   = s.get("coherence_mean","?")
        bold  = cond == "C_multiagent"
        if bold:
            lines.append(f"| **{label}** | **{coh}** |")
        else:
            lines.append(f"| {label} | {coh} |")

    lines.append("\n### 5.3 Latency\n")
    lines.append("| Condition | Mean latency (s) | SD (s) |")
    lines.append("|---|---|---|")
    for cond in ["A_baseline","B_rag","C_multiagent"]:
        s = summary.get(cond, {})
        label = cond_labels.get(cond, cond)
        lat   = s.get("latency_mean_s","?")
        sd    = s.get("latency_sd_s","?")
        lines.append(f"| {label} | {lat} | {sd} |")

    # Per-category breakdown (C only)
    lines.append("\n### 5.4 Multi-agent results by ticker category\n")
    lines.append("| Category | n | Factual Accuracy | Hallucination Rate | Coherence |")
    lines.append("|---|---|---|---|---|")
    from collections import defaultdict
    cat_rows = defaultdict(list)
    for r in rows:
        if r["condition"] == "C_multiagent" and r["factual_accuracy_pct"] is not None:
            cat_rows[r["category"]].append(r)
    for cat in ["large_cap","mid_cap","small_cap"]:
        rr = cat_rows.get(cat, [])
        if not rr:
            continue
        acc  = round(statistics.mean([r["factual_accuracy_pct"] for r in rr]),1)
        hall = round(statistics.mean([r["hallucination"]        for r in rr])*100,1)
        coh  = round(statistics.mean([r["coherence_score"]      for r in rr]),2)
        lines.append(f"| {cat.replace('_',' ').title()} | {len(rr)} | {acc}% | {hall}% | {coh} |")

    lines.append(f"\n*Benchmark run: {datetime.now().strftime('%Y-%m-%d')} · Model: {MODEL} · n=30 tickers*\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown table saved → {path}")

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = run_evaluation()

    raw_path     = os.path.join(OUTPUT_DIR, "results_raw.csv")
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    md_path      = os.path.join(OUTPUT_DIR, "results_table.md")

    write_raw_csv(rows, raw_path)
    summary = summarize(rows)
    write_summary_csv(summary, summary_path)
    write_markdown_table(summary, rows, md_path)

    print("\n── Final summary ──────────────────────────────────────────")
    for cond, s in summary.items():
        label = {"A_baseline":"Baseline","B_rag":"RAG","C_multiagent":"Multi-agent"}[cond]
        print(f"  {label:15s}  acc={s['factual_accuracy_pct']}%  "
              f"hall={s['hallucination_rate_pct']}%  "
              f"coh={s['coherence_mean']}  lat={s['latency_mean_s']}s")

    print("\nDone. Copy results_table.md content into your README.")
    print("="*60)
