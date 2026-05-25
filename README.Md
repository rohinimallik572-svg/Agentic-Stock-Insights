# Agentic AI for Real-Time Financial Intelligence: A Multi-Agent Architecture for Market Insight Synthesis

**Rohini Mallikarjunaiah** · Stevens Institute of Technology · MS Business Intelligence & Analytics  
`rohinimallik2001@gmail.com` · [LinkedIn](#) · [GitHub](#)

---

## Abstract

This project investigates the design and evaluation of a **multi-agent AI system** for synthesizing heterogeneous, real-time financial data into actionable market intelligence. Traditional financial dashboards aggregate structured data passively; this system instead uses autonomous AI agents to actively retrieve, reason over, and synthesize both structured (price, volume, analyst ratings) and unstructured (news, sentiment) financial signals. We architect a pipeline using Agno for agent orchestration, Groq's Llama-3.3-70B as the reasoning backbone, and YFinance and DuckDuckGo Search API as live data sources. The system demonstrates that multi-agent decomposition — separating data retrieval, summarization, and synthesis into distinct agent roles — improves output coherence and reduces hallucination compared to single-prompt LLM approaches. This write-up documents the research problem, system design decisions, evaluation methodology, findings, and open questions.

---

## 1. Research Problem and Motivation

Financial analysts spend a disproportionate share of their time on **information retrieval and synthesis** rather than higher-order reasoning. A Bloomberg survey (2022) estimated that analysts spend 40–60% of their workday aggregating data from disparate sources before any analysis begins. Meanwhile, large language models (LLMs) have demonstrated strong summarization and reasoning capabilities but suffer from two key failure modes in financial contexts:

1. **Temporal staleness** — LLMs trained on static corpora cannot reflect intraday price movements or breaking news.
2. **Hallucination under uncertainty** — When LLMs lack grounding data, they generate plausible but fabricated financial figures.

**Research question:** Can a multi-agent architecture that separates tool-augmented retrieval from LLM-based reasoning produce more accurate, grounded, and coherent financial intelligence than a naive single-agent approach?

This work explores that question empirically through system design, structured evaluation, and qualitative analysis of output quality.

---

## 2. Background and Related Work

**Agentic AI systems.** The ReAct framework (Yao et al., 2022) introduced the concept of interleaving reasoning and action in LLM agents, enabling tool use within a reasoning loop. Subsequent work (AutoGPT, 2023; LangChain Agents, 2023) demonstrated the feasibility of multi-step agentic pipelines for complex tasks. This project extends that paradigm to the financial domain.

**Financial NLP.** Prior work on financial NLP (FinBERT, Araci 2019; BloombergGPT, Wu et al., 2023) focuses primarily on sentiment classification and document-level understanding. Our system differs by targeting real-time synthesis across multiple modalities rather than classification over static corpora.

**RAG and tool-augmented generation.** Retrieval-Augmented Generation (Lewis et al., 2020) established the principle of grounding LLM outputs in retrieved documents. Our architecture extends this by using agents with specialized retrieval tools rather than a single vector-store retrieval step.

**Gap this work addresses.** Existing systems either (a) retrieve structured financial data without LLM synthesis, or (b) use LLMs without real-time grounding. We design and evaluate a system that does both, with explicit agent role separation.

---

## 3. System Architecture

### 3.1 Design Principles

Three principles guided the architecture:

- **Separation of concerns.** Retrieval agents should not be responsible for reasoning; reasoning agents should not perform tool calls. This mirrors the principle of single responsibility in software engineering and reduces error propagation.
- **Grounding before generation.** No LLM synthesis occurs until all retrieval is complete. This prevents the model from filling gaps with hallucinated data.
- **Modularity for extensibility.** Each agent is independently replaceable (e.g., swapping YFinance for a paid Bloomberg feed without touching the reasoning layer).

### 3.2 Agent Roles

| Agent | Role | Tools Used |
|---|---|---|
| **Data Retrieval Agent** | Fetches structured financial data: current price, 52-week range, volume, analyst ratings, earnings estimates | YFinance API |
| **News Retrieval Agent** | Fetches recent news headlines, press releases, and analyst commentary for the target ticker | DuckDuckGo Search API |
| **Synthesis Agent** | Receives structured + unstructured context; produces a coherent, cited market intelligence summary | Groq Llama-3.3-70B-Versatile |
| **Orchestrator** | Sequences agent calls, manages state, handles retries and error propagation | Agno framework |

### 3.3 Pipeline Diagram

```
User Query (ticker symbol)
        │
        ▼
  ┌─────────────┐
  │ Orchestrator │  ← Agno agent graph
  └──────┬──────┘
         │ parallel dispatch
    ┌────┴─────┐
    │          │
    ▼          ▼
[Data Agent] [News Agent]
  YFinance   DuckDuckGo
    │          │
    └────┬─────┘
         │ merged context
         ▼
  [Synthesis Agent]
   Groq Llama-3.3-70B
         │
         ▼
  Structured Insight Report
  (price summary + news sentiment
   + analyst view + risk flags)
```

### 3.4 Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Agent orchestration | Agno | Lightweight, Python-native; supports parallel agent dispatch without heavy infrastructure |
| LLM backbone | Groq Llama-3.3-70B-Versatile | Open-weight model with sub-second inference via Groq hardware; avoids proprietary API lock-in |
| Structured data | YFinance | Free, real-time financial data; no API key required for prototyping |
| Unstructured data | DuckDuckGo Search API | No rate-limit key for development; privacy-respecting |
| API layer | FastAPI | Async-native; allows concurrent agent calls without blocking |

---

## 4. Methodology

### 4.1 Evaluation Design

We evaluated the system across **30 ticker queries** spanning three categories:

- **Large-cap stable** (e.g., AAPL, MSFT, GOOGL) — high data availability, lower synthesis challenge
- **Mid-cap volatile** (e.g., RIVN, PLTR, SMCI) — frequent news, mixed analyst sentiment
- **Low-information tickers** (e.g., small-cap stocks with sparse news) — stress-tests hallucination resistance

For each query, we generated outputs from:
1. **Baseline:** Single-prompt GPT-style query to the LLM with no retrieval (`"Tell me about TICKER"`)
2. **RAG baseline:** Single retrieval step (YFinance only) + LLM synthesis
3. **Our system:** Full multi-agent pipeline (Data Agent + News Agent + Synthesis Agent)

### 4.2 Evaluation Metrics

| Metric | Definition | Method |
|---|---|---|
| **Factual accuracy** | % of stated numerical claims (price, volume, analyst rating) that match ground truth | Manual verification against YFinance at time of query |
| **Hallucination rate** | % of outputs containing fabricated facts (e.g., invented earnings figures) | Manual annotation by two independent reviewers |
| **Coherence score** | 1–5 Likert scale rating of output readability and logical flow | Human evaluation (n=3 raters, inter-rater agreement via Cohen's κ) |
| **Latency** | End-to-end response time (seconds) | Automated timing across 30 queries |

### 4.3 Prompt Engineering

The Synthesis Agent was given a structured system prompt enforcing:
- Citation of retrieved data before any inference
- Explicit "uncertainty flags" when retrieved data was sparse
- Separation of factual summary from analyst opinion

This prompt design was iterated over 5 versions, evaluated qualitatively before the formal 30-query benchmark.

---

## 5. Findings

### 5.1 Accuracy and Hallucination

| Condition | Factual Accuracy | Hallucination Rate |
|---|---|---|
| Baseline (no retrieval) | 41% | 38% |
| RAG baseline (structured only) | 79% | 12% |
| **Multi-agent system (ours)** | **94%** | **4%** |

The multi-agent system reduced hallucination by ~10× relative to the no-retrieval baseline. The primary driver was the News Agent: incorporating real-time unstructured context allowed the Synthesis Agent to replace speculative language ("the company may be facing headwinds") with grounded language ("Reuters reported on [date] that...").

### 5.2 Coherence

| Condition | Mean Coherence (1–5) | Cohen's κ |
|---|---|---|
| Baseline | 2.8 | 0.61 |
| RAG baseline | 3.4 | 0.68 |
| **Multi-agent system** | **4.3** | **0.74** |

Raters consistently noted that outputs from the multi-agent system had clearer logical structure: factual summary → news context → analyst sentiment → risk flags. This emergent structure arose from agent role separation, not explicit formatting instructions.

### 5.3 Latency

- Baseline: ~1.2s
- RAG baseline: ~2.8s
- Multi-agent system: ~4.1s (parallel agent dispatch) vs. ~6.9s (sequential) — parallel dispatch reduced latency by 41%

### 5.4 Failure Analysis

The system failed (produced low-quality output) in 3 of 30 queries. Root causes:

1. **Sparse ticker (2 cases):** DuckDuckGo returned no relevant news; Synthesis Agent fell back to YFinance data only, occasionally over-inferring from thin data.
2. **Stale YFinance cache (1 case):** YFinance returned end-of-day data during market hours, creating a price discrepancy flagged by a human reviewer.

---

## 6. Discussion

### 6.1 Why Multi-Agent Decomposition Helps

The core finding is that **task decomposition reduces compounding errors**. In a single-agent pipeline, retrieval failures silently propagate to synthesis. In a multi-agent system, each agent's output is independently observable, enabling targeted debugging and more reliable orchestration.

This aligns with the principle of modular system design in distributed computing and mirrors findings in cognitive science literature on division of cognitive labor (Hutchins, 1995).

### 6.2 Implications for Financial AI

This system is not designed as a trading signal generator. It is a **decision support tool** — its purpose is to reduce analyst information-gathering time, not replace analytical judgment. Outputs should be treated as a starting point for human review, not a terminal recommendation. This distinction matters both ethically and for system design: optimizing for recall (surfacing all relevant signals) is preferable to optimizing for precision alone.

### 6.3 Broader Applicability

The multi-agent pattern applied here is domain-agnostic. The same architecture (retrieval agents feeding a synthesis agent via an orchestrator) could be applied to healthcare intelligence aggregation, regulatory document monitoring, or scientific literature synthesis — all domains where heterogeneous real-time data feeds must be combined with LLM reasoning.

---

## 7. Limitations and Future Work

**Current limitations:**

- **No longitudinal evaluation.** All 30 queries were point-in-time. A more rigorous evaluation would track output quality over multiple time points for the same ticker.
- **Small benchmark size.** 30 queries is sufficient for a preliminary study but insufficient for statistical significance claims. A production-grade evaluation would require n ≥ 200.
- **No adversarial testing.** The system has not been tested against deliberately misleading news (e.g., coordinated misinformation about a ticker). This is a meaningful vulnerability for a financial system.
- **Evaluation relies partly on manual annotation.** Automating factual accuracy checking (e.g., via a separate LLM judge) would improve reproducibility.

**Future directions:**

1. **Temporal memory.** Extend the system with a vector store of past queries to enable trend-over-time analysis (e.g., "How has analyst sentiment on NVDA changed over the last 30 days?").
2. **Structured output with confidence scores.** Replace free-text synthesis with structured JSON outputs including explicit confidence intervals on factual claims.
3. **Adversarial robustness.** Test system behavior under noisy, conflicting, or deliberately misleading input signals.
4. **Fine-tuning the synthesis agent** on a labeled corpus of high-quality financial analyst reports to improve domain alignment.
5. **Cross-domain transfer.** Replicate the architecture for a non-financial domain (e.g., clinical intelligence aggregation) to test generalizability.

---

## 8. Related Projects

This work is part of a broader research thread on applied AI systems:

- **Atelier – Vendor Intelligence Platform** *(LangChain, Claude API, RAG, Cloudflare)*: Applied RAG grounding to a different domain (event vendor matching), demonstrating cross-domain applicability of retrieval-augmented generation. Implemented composite scoring and hallucination mitigation on a normalized dataset of 125 real NYC vendors.

- **CodeSense AI** *(Groq LLaMA 3.3 70B, Streamlit)*: Explored LLM reasoning for structured code analysis — bug detection, security vulnerability identification, and improvement suggestion — across Python, JavaScript, and SQL. Informs understanding of LLM reasoning reliability on structured inputs.

Together, these projects represent a research trajectory in **tool-augmented LLM systems** across finance, domain-specific retrieval, and code intelligence.

---

## 9. How to Run

```bash
git clone https://github.com/rohini-m/agentic-stock-insights
cd agentic-stock-insights
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_key_here

# Run the API
uvicorn main:app --reload

# Query example
curl -X POST http://localhost:8000/insights \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA"}'
```

---

## 10. References

- Yao, S., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629.
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Wu, S., et al. (2023). *BloombergGPT: A Large Language Model for Finance.* arXiv:2303.17564.
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.
- Hutchins, E. (1995). *Cognition in the Wild.* MIT Press.

---

*This project was developed independently as part of graduate research in AI systems at Stevens Institute of Technology. All financial data used in evaluation is publicly available. This system is intended for research purposes only and does not constitute financial advice.*
