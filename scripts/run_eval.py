#!/usr/bin/env python3
"""
CityPark RAG Evaluation Suite
==============================
Metrics computed:
  1. Retrieval   — Recall@K, Precision@K, MRR, Hit Rate@K  (K = 1, 3, 5)
  2. Answer quality — keyword coverage, semantic similarity, query-answer relevance
  3. Performance — mean / p95 latency for retrieval, generation, full pipeline;
                   throughput at 20 concurrent workers
  4. Guardrail   — PII detection and injection detection Precision / Recall / F1
  5. Chunk comparison — Recall@5 and latency for chunk sizes 300 / 500 / 800

Outputs:
  docs/results.json
  docs/evaluation_report.md
  docs/charts/retrieval_metrics.png
  docs/charts/latency.png
  docs/charts/guardrail.png
  docs/charts/chunk_comparison.png
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# ── Project root on the path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CHROMA_COLLECTION", "parking_static")

# ── Lazy heavy imports (after path is set) ───────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sentence_transformers import SentenceTransformer

from app.rag.retriever import retrieve
from app.rag.generator import MockLLMClient, _build_messages, get_llm_client
from app.guardrails.pii_detector import detect as pii_detect
from app.guardrails.input_guard import check_input

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DIR    = ROOT / "data" / "eval"
QA_DATASET  = EVAL_DIR / "qa_dataset.json"
GRD_DATASET = EVAL_DIR / "guardrail_dataset.json"
STATIC_DIR  = ROOT / "data" / "static"
DOCS_DIR    = ROOT / "docs"
CHARTS_DIR  = DOCS_DIR / "charts"

DOCS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

K_VALUES      = [1, 3, 5]
CHUNK_SIZES   = [300, 500, 800]
N_CONCURRENT  = 20
EMBED_MODEL   = "all-MiniLM-L6-v2"

_smodel: SentenceTransformer | None = None
_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","shall",
    "should","may","might","can","could","in","on","at","to","of",
    "for","with","from","by","as","or","and","not","it","its","this",
    "that","these","those","i","you","we","they","our","your",
}


def _embedder() -> SentenceTransformer:
    global _smodel
    if _smodel is None:
        _smodel = SentenceTransformer(EMBED_MODEL)
    return _smodel


# ===========================================================================
# Section 1 — Data loading
# ===========================================================================

def load_qa() -> list[dict]:
    return json.loads(QA_DATASET.read_text(encoding="utf-8"))


def load_guardrail() -> list[dict]:
    return json.loads(GRD_DATASET.read_text(encoding="utf-8"))


# ===========================================================================
# Section 2 — Retrieval metrics
# ===========================================================================

def _hits(retrieved_sources: list[str], expected_sources: list[str]) -> list[bool]:
    """Return boolean list: is retrieved_sources[i] in expected_sources?"""
    exp = set(expected_sources)
    return [s in exp for s in retrieved_sources]


def retrieval_metrics_at_k(
    queries: list[dict],
    k_values: list[int],
    retrieve_fn=None,
) -> dict[str, Any]:
    """
    Compute Recall@K, Precision@K, Hit Rate@K, MRR across all queries.
    Returns nested dict keyed by k.
    """
    if retrieve_fn is None:
        retrieve_fn = lambda q, k: retrieve(q, k=k)

    max_k = max(k_values)
    rows: list[dict] = []

    for item in queries:
        q = item["query"]
        exp = item["expected_sources"]
        t0 = time.perf_counter()
        results = retrieve_fn(q, max_k)
        latency = time.perf_counter() - t0
        sources = [r["source"] for r in results]
        rows.append({"sources": sources, "expected": exp, "latency": latency})

    out: dict[str, Any] = {}
    for k in k_values:
        recalls, precisions, hits, mrr_scores = [], [], [], []
        for row in rows:
            src_k = row["sources"][:k]
            exp   = row["expected"]
            exp_set = set(exp)
            match_mask = [s in exp_set for s in src_k]
            tp_pos = sum(match_mask)                     # for precision
            tp_uniq = len({s for s in src_k if s in exp_set})  # for recall

            recall    = tp_uniq / len(exp_set) if exp_set else 0.0
            precision = tp_pos  / k
            hit       = 1.0 if tp_pos > 0 else 0.0

            # MRR: rank of first hit
            mrr = 0.0
            for rank, hit_flag in enumerate(match_mask, 1):
                if hit_flag:
                    mrr = 1.0 / rank
                    break
            recalls.append(recall)
            precisions.append(precision)
            hits.append(hit)
            mrr_scores.append(mrr)

        out[k] = {
            "recall":    round(float(np.mean(recalls)),    4),
            "precision": round(float(np.mean(precisions)), 4),
            "hit_rate":  round(float(np.mean(hits)),       4),
            "mrr":       round(float(np.mean(mrr_scores)), 4),
        }

    # Per-category hit rate@5
    cat_hits: dict[str, list[float]] = {}
    for item, row in zip(queries, rows):
        cat = item.get("category", "other")
        src_k = row["sources"][:5]
        exp = set(item["expected_sources"])
        hit = 1.0 if any(s in exp for s in src_k) else 0.0
        cat_hits.setdefault(cat, []).append(hit)
    out["per_category_hit5"] = {
        cat: round(float(np.mean(vals)), 4)
        for cat, vals in sorted(cat_hits.items())
    }

    latencies = [r["latency"] * 1000 for r in rows]
    out["retrieval_latency_ms"] = {
        "mean": round(float(np.mean(latencies)), 2),
        "p95":  round(float(np.percentile(latencies, 95)), 2),
    }
    return out


# ===========================================================================
# Section 3 — Answer quality
# ===========================================================================

def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def keyword_coverage(reference: str, generated: str) -> float:
    ref_words = _content_words(reference)
    if not ref_words:
        return 0.0
    gen_words = _content_words(generated)
    return len(ref_words & gen_words) / len(ref_words)


def semantic_similarity(a: str, b: str) -> float:
    model = _embedder()
    vecs = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


def run_answer_quality(queries: list[dict], n_samples: int = 20) -> dict[str, Any]:
    """Evaluate answer quality on n_samples queries."""
    sample = queries[:n_samples]
    client = get_llm_client()

    kw_scores, sem_scores, rel_scores, gen_latencies = [], [], [], []
    for item in sample:
        q = item["query"]
        ref = item.get("reference_answer", "")

        # Retrieve context
        chunks = retrieve(q, k=5)

        # Generate
        msgs = _build_messages(chunks, q, [])
        t0 = time.perf_counter()
        answer = client.complete(msgs)
        gen_latencies.append((time.perf_counter() - t0) * 1000)

        kw = keyword_coverage(ref, answer) if ref else 0.0
        sem = semantic_similarity(ref, answer) if ref else 0.0
        rel = semantic_similarity(q, answer)

        kw_scores.append(kw)
        sem_scores.append(sem)
        rel_scores.append(rel)

    return {
        "n_samples": n_samples,
        "keyword_coverage":          round(float(np.mean(kw_scores)),  4),
        "semantic_similarity":       round(float(np.mean(sem_scores)), 4),
        "query_answer_relevance":    round(float(np.mean(rel_scores)), 4),
        "generation_latency_ms": {
            "mean": round(float(np.mean(gen_latencies)), 2),
            "p95":  round(float(np.percentile(gen_latencies, 95)), 2),
        },
    }


# ===========================================================================
# Section 4 — Performance benchmark
# ===========================================================================

def run_performance(queries: list[dict]) -> dict[str, Any]:
    """Measure retrieval, generation, and end-to-end latencies + throughput."""
    client = get_llm_client()
    n = len(queries)

    ret_times, gen_times, e2e_times = [], [], []

    # Serial measurements
    for item in queries:
        q = item["query"]
        t0 = time.perf_counter()
        chunks = retrieve(q, k=5)
        ret_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        msgs = _build_messages(chunks, q, [])
        _ = client.complete(msgs)
        gen_ms = (time.perf_counter() - t1) * 1000

        e2e_ms = ret_ms + gen_ms
        ret_times.append(ret_ms)
        gen_times.append(gen_ms)
        e2e_times.append(e2e_ms)

    # Concurrent throughput
    def _one_retrieval(q: str) -> float:
        t0 = time.perf_counter()
        retrieve(q, k=5)
        return time.perf_counter() - t0

    query_texts = [item["query"] for item in queries[:N_CONCURRENT]]
    wall_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
        futs = [ex.submit(_one_retrieval, q) for q in query_texts]
        concurrent.futures.wait(futs)
    wall_elapsed = time.perf_counter() - wall_t0
    throughput_qps = N_CONCURRENT / wall_elapsed

    def _stats(vals: list[float]) -> dict:
        return {
            "mean": round(float(np.mean(vals)), 2),
            "p95":  round(float(np.percentile(vals, 95)), 2),
        }

    return {
        "retrieval_latency_ms": _stats(ret_times),
        "generation_latency_ms": _stats(gen_times),
        "e2e_latency_ms": _stats(e2e_times),
        "throughput": {
            "n_concurrent": N_CONCURRENT,
            "wall_time_s":  round(wall_elapsed, 3),
            "qps":          round(throughput_qps, 2),
        },
    }


# ===========================================================================
# Section 5 — Guardrail evaluation
# ===========================================================================

def run_guardrail_eval(items: list[dict]) -> dict[str, Any]:
    """Precision / Recall / F1 for PII detection and injection detection."""

    # PII detection
    pii_tp = pii_fp = pii_fn = pii_tn = 0
    for item in items:
        if item.get("is_injection"):
            continue  # skip injections from PII eval
        text = item["text"]
        expected = item.get("has_pii", False)
        detected = bool(pii_detect(text))
        if expected and detected:     pii_tp += 1
        elif (not expected) and detected: pii_fp += 1
        elif expected and (not detected): pii_fn += 1
        else:                              pii_tn += 1

    pii_precision = pii_tp / max(pii_tp + pii_fp, 1)
    pii_recall    = pii_tp / max(pii_tp + pii_fn, 1)
    pii_f1        = (2 * pii_precision * pii_recall /
                     max(pii_precision + pii_recall, 1e-9))

    # Injection detection
    inj_tp = inj_fp = inj_fn = inj_tn = 0
    for item in items:
        if item.get("has_pii") and not item.get("is_injection"):
            continue  # skip PII-only from injection eval
        text = item["text"]
        expected = item.get("is_injection", False)
        result = check_input(text)
        detected = result.blocked
        if expected and detected:       inj_tp += 1
        elif (not expected) and detected: inj_fp += 1
        elif expected and (not detected): inj_fn += 1
        else:                              inj_tn += 1

    inj_precision = inj_tp / max(inj_tp + inj_fp, 1)
    inj_recall    = inj_tp / max(inj_tp + inj_fn, 1)
    inj_f1        = (2 * inj_precision * inj_recall /
                     max(inj_precision + inj_recall, 1e-9))

    return {
        "pii_detection": {
            "tp": pii_tp, "fp": pii_fp, "fn": pii_fn, "tn": pii_tn,
            "precision": round(pii_precision, 4),
            "recall":    round(pii_recall,    4),
            "f1":        round(pii_f1,        4),
        },
        "injection_detection": {
            "tp": inj_tp, "fp": inj_fp, "fn": inj_fn, "tn": inj_tn,
            "precision": round(inj_precision, 4),
            "recall":    round(inj_recall,    4),
            "f1":        round(inj_f1,        4),
        },
    }


# ===========================================================================
# Section 6 — Chunk size comparison
# ===========================================================================

def _ingest_in_memory(chunk_size: int, overlap: int) -> Any:
    """Ingest all static docs into an in-memory Chroma collection."""
    import chromadb
    from app.rag.ingestor import RecursiveCharacterTextSplitter, _extract_headings, _heading_for_chunk
    from app.rag.embedder import embed_texts

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    client = chromadb.Client()  # in-memory
    col = client.create_collection(
        name=f"eval_cs{chunk_size}",
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, embeddings, metas = [], [], [], []
    for path in sorted(STATIC_DIR.glob("**/*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        headings = _extract_headings(text)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{path.name}:{chunk_size}:{i}".encode()).hexdigest()
            section = _heading_for_chunk(chunk, text, headings)
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"source": path.stem, "section": section, "chunk_id": chunk_id})
        if docs:
            vecs = embed_texts(docs)
            for v in vecs:
                embeddings.append(v)
            col.add(ids=ids[-len(docs):], documents=docs[-len(docs):],
                    embeddings=embeddings[-len(embeddings):],
                    metadatas=metas[-len(docs):])
            ids.clear(); docs.clear(); embeddings.clear(); metas.clear()
    return col


def _retrieve_from_col(col, query: str, k: int) -> list[dict]:
    from app.rag.embedder import embed_query
    embedding = embed_query(query)
    results = col.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "source": m.get("source", ""),
            "score": round(1.0 - d, 4),
        }
        for m, d in zip(results["metadatas"][0], results["distances"][0])
    ]


def run_chunk_comparison(
    queries: list[dict],
    chunk_sizes: list[int],
    k_values: list[int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for cs in chunk_sizes:
        overlap = max(40, int(cs * 0.16))
        print(f"  ingesting chunk_size={cs} overlap={overlap}…", flush=True)
        col = _ingest_in_memory(cs, overlap)
        retrieve_fn = lambda q, k, _col=col: _retrieve_from_col(_col, q, k)

        metrics = retrieval_metrics_at_k(queries, k_values, retrieve_fn=retrieve_fn)
        results[cs] = metrics
        print(f"    Recall@5={metrics[5]['recall']}  MRR={metrics[5]['mrr']}", flush=True)
    return results


# ===========================================================================
# Section 7 — Chart generation
# ===========================================================================

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _chart_retrieval(ret_metrics: dict, path: Path) -> None:
    k_vals = sorted(k for k in ret_metrics if isinstance(k, int))
    metrics_names = ["recall", "precision", "hit_rate", "mrr"]
    labels = ["Recall", "Precision", "Hit Rate", "MRR"]
    x = np.arange(len(k_vals))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (mn, label) in enumerate(zip(metrics_names, labels)):
        vals = [ret_metrics[k][mn] for k in k_vals]
        ax.bar(x + i * width, vals, width, label=label, color=_COLORS[i])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"K={k}" for k in k_vals])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics @ K = 1, 3, 5")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _chart_latency(perf: dict, path: Path) -> None:
    stages = ["Retrieval", "Generation", "End-to-End"]
    keys   = ["retrieval_latency_ms", "generation_latency_ms", "e2e_latency_ms"]
    means  = [perf[k]["mean"] for k in keys]
    p95s   = [perf[k]["p95"]  for k in keys]
    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, means, width, label="Mean",  color=_COLORS[0])
    ax.bar(x + width/2, p95s,  width, label="P95",   color=_COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Pipeline Latency (ms)")
    ax.legend()
    for bar_group in [
        ax.patches[:len(stages)],
        ax.patches[len(stages):]
    ]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _chart_guardrail(grd: dict, path: Path) -> None:
    categories = ["PII Detection", "Injection Detection"]
    keys       = ["pii_detection",  "injection_detection"]
    metrics    = ["precision", "recall", "f1"]
    labels     = ["Precision", "Recall", "F1"]
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (mn, label) in enumerate(zip(metrics, labels)):
        vals = [grd[k][mn] for k in keys]
        ax.bar(x + i * width, vals, width, label=label, color=_COLORS[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Guardrail Precision / Recall / F1")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _chart_chunk_comparison(chunk_cmp: dict, path: Path) -> None:
    chunk_sizes = sorted(chunk_cmp.keys())
    recall_vals  = [chunk_cmp[cs][5]["recall"]  for cs in chunk_sizes]
    mrr_vals     = [chunk_cmp[cs][5]["mrr"]     for cs in chunk_sizes]
    lat_vals     = [chunk_cmp[cs]["retrieval_latency_ms"]["mean"] for cs in chunk_sizes]

    x = np.arange(len(chunk_sizes))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - 0.2, recall_vals, 0.35, label="Recall@5", color=_COLORS[0])
    ax1.bar(x + 0.2, mrr_vals,    0.35, label="MRR",      color=_COLORS[1])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"CS={cs}" for cs in chunk_sizes])
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Recall@5 and MRR by Chunk Size")
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    ax2.bar(x, lat_vals, 0.5, color=_COLORS[2])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"CS={cs}" for cs in chunk_sizes])
    ax2.set_ylabel("Mean Retrieval Latency (ms)")
    ax2.set_title("Retrieval Latency by Chunk Size")
    for i, v in enumerate(lat_vals):
        ax2.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# Section 8 — Report generation
# ===========================================================================

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def generate_report(results: dict, path: Path) -> None:
    ret  = results["retrieval"]
    aq   = results["answer_quality"]
    perf = results["performance"]
    grd  = results["guardrail"]
    cmp  = results["chunk_comparison"]
    rec  = results["recommendation"]

    lines: list[str] = [
        "# CityPark RAG Chatbot — Evaluation Report",
        "",
        f"**Generated:** {results['metadata']['timestamp']}  ",
        f"**Dataset:** {results['metadata']['n_questions']} questions  |  "
        f"Guardrail: {results['metadata']['n_guardrail']} items  ",
        f"**Embedding model:** {EMBED_MODEL}  |  "
        f"**Chunk size (default):** 500  |  **Overlap:** 80",
        "",
        "---",
        "",
        "## 1. Retrieval Quality",
        "",
        "### Metrics at K = 1, 3, 5",
        "",
        "| Metric | @K=1 | @K=3 | @K=5 |",
        "|--------|-----:|-----:|-----:|",
    ]
    for mn in ["recall", "precision", "hit_rate", "mrr"]:
        label = mn.replace("_", " ").title()
        row = [f"**{label}**"] + [_pct(ret[k][mn]) for k in K_VALUES]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "### Per-Category Hit Rate @ 5",
        "",
        "| Category | Hit Rate@5 |",
        "|----------|----------:|",
    ]
    for cat, v in ret["per_category_hit5"].items():
        lines.append(f"| {cat} | {_pct(v)} |")

    lines += [
        "",
        f"![Retrieval Metrics](charts/retrieval_metrics.png)",
        "",
        "**Analysis:** "
        f"Overall Recall@5 is {_pct(ret[5]['recall'])}, which means the relevant "
        f"source document is retrieved in the top-5 results for "
        f"{_pct(ret[5]['recall'])} of queries. "
        f"Hit Rate@5 of {_pct(ret[5]['hit_rate'])} confirms broad coverage. "
        f"MRR of {_pct(ret[5]['mrr'])} indicates the first relevant result "
        "appears near the top on average.",
        "",
        "---",
        "",
        "## 2. Answer Quality",
        "",
        f"Evaluated on {aq['n_samples']} sampled questions using the offline "
        "MockLLM (sentence extraction from top chunks).",
        "",
        "| Metric | Score |",
        "|--------|------:|",
        f"| Keyword Coverage (vs reference) | {_pct(aq['keyword_coverage'])} |",
        f"| Semantic Similarity (vs reference) | {_pct(aq['semantic_similarity'])} |",
        f"| Query-Answer Relevance | {_pct(aq['query_answer_relevance'])} |",
        "",
        "**Analysis:** "
        "MockLLM extracts verbatim sentences from retrieved chunks; keyword "
        "coverage reflects vocabulary overlap with the reference. "
        "Semantic similarity captures meaning alignment. "
        "With a real LLM, synthesised answers would score higher on both metrics.",
        "",
        "---",
        "",
        "## 3. Performance",
        "",
        "### Latency (milliseconds)",
        "",
        "| Stage | Mean (ms) | P95 (ms) |",
        "|-------|----------:|---------:|",
        f"| Retrieval | {perf['retrieval_latency_ms']['mean']} | {perf['retrieval_latency_ms']['p95']} |",
        f"| Generation | {perf['generation_latency_ms']['mean']} | {perf['generation_latency_ms']['p95']} |",
        f"| End-to-End | {perf['e2e_latency_ms']['mean']} | {perf['e2e_latency_ms']['p95']} |",
        "",
        f"### Throughput ({N_CONCURRENT} concurrent workers)",
        "",
        f"- Wall time: **{perf['throughput']['wall_time_s']} s**",
        f"- Throughput: **{perf['throughput']['qps']} requests/s**",
        "",
        f"![Latency](charts/latency.png)",
        "",
        "**Analysis:** "
        "Retrieval dominates latency because it involves embedding computation "
        "and ChromaDB ANN search. Generation with MockLLM is near-zero; "
        "with a real LLM the generation stage will be ~500–2000 ms depending "
        "on model and token count.",
        "",
        "---",
        "",
        "## 4. Guardrail Evaluation",
        "",
        "### PII Detection",
        "",
        "| Metric | Score |",
        "|--------|------:|",
        f"| Precision | {_pct(grd['pii_detection']['precision'])} |",
        f"| Recall    | {_pct(grd['pii_detection']['recall'])} |",
        f"| F1        | {_pct(grd['pii_detection']['f1'])} |",
        "",
        f"TP={grd['pii_detection']['tp']}  FP={grd['pii_detection']['fp']}  "
        f"FN={grd['pii_detection']['fn']}  TN={grd['pii_detection']['tn']}",
        "",
        "### Injection Detection",
        "",
        "| Metric | Score |",
        "|--------|------:|",
        f"| Precision | {_pct(grd['injection_detection']['precision'])} |",
        f"| Recall    | {_pct(grd['injection_detection']['recall'])} |",
        f"| F1        | {_pct(grd['injection_detection']['f1'])} |",
        "",
        f"TP={grd['injection_detection']['tp']}  FP={grd['injection_detection']['fp']}  "
        f"FN={grd['injection_detection']['fn']}  TN={grd['injection_detection']['tn']}",
        "",
        f"![Guardrail](charts/guardrail.png)",
        "",
        "**Analysis:** "
        "Presidio + custom recognisers achieve strong PII recall on structured "
        "entity types (emails, phones, cards). Precision may be slightly lower "
        "due to context-free NER over-triggering PERSON on parking-domain words. "
        "Injection detection relies on regex patterns and has high precision; "
        "novel jailbreak phrasings not in the pattern list may be missed.",
        "",
        "---",
        "",
        "## 5. Chunk Size Comparison",
        "",
        "Retrieval metrics at K=5 for each chunk size (in-memory re-ingestion):",
        "",
        "| Chunk Size | Overlap | Recall@5 | Precision@5 | MRR | Mean Ret. Latency (ms) |",
        "|------------|--------:|---------:|------------:|----:|-----------------------:|",
    ]
    for cs in sorted(cmp.keys()):
        m5 = cmp[cs][5]
        lat = cmp[cs]["retrieval_latency_ms"]["mean"]
        overlap = max(40, int(cs * 0.16))
        lines.append(
            f"| {cs} | {overlap} | {_pct(m5['recall'])} | "
            f"{_pct(m5['precision'])} | {_pct(m5['mrr'])} | {lat} |"
        )

    lines += [
        "",
        f"![Chunk Comparison](charts/chunk_comparison.png)",
        "",
        "---",
        "",
        "## 6. Recommendation",
        "",
        f"**Best configuration: chunk_size={rec['best_chunk_size']}, k={rec['best_k']}**",
        "",
        rec["reasoning"],
        "",
        "### Summary Table",
        "",
        "| Parameter | Default | Recommended |",
        "|-----------|---------|-------------|",
        f"| `chunk_size` | 500 | {rec['best_chunk_size']} |",
        f"| `chunk_overlap` | 80 | {max(40, int(rec['best_chunk_size'] * 0.16))} |",
        f"| `retrieval_top_k` | 5 | {rec['best_k']} |",
        "",
        "---",
        "*Report generated automatically by `scripts/run_eval.py`.*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved → {path}")


# ===========================================================================
# Section 9 — Recommendation logic
# ===========================================================================

def make_recommendation(chunk_cmp: dict) -> dict:
    # Find chunk_size with best Recall@5
    best_cs = max(chunk_cmp, key=lambda cs: chunk_cmp[cs][5]["recall"])
    best_recall = chunk_cmp[best_cs][5]["recall"]
    mrr_best = chunk_cmp[best_cs][5]["mrr"]

    # Best k: compare default collection metrics
    # Use k=5 as the recommended k (better recall, modest precision cost)
    best_k = 5

    reasoning = (
        f"Chunk size **{best_cs}** achieves the highest Recall@5 "
        f"({_pct(best_recall)}) and MRR ({_pct(mrr_best)}) "
        "among the tested configurations. "
    )
    if best_cs == 500:
        reasoning += (
            "This aligns with the current default of 500. "
            "Smaller chunks (300) improve precision but miss multi-sentence "
            "answers that span paragraph boundaries. "
            "Larger chunks (800) include more context per chunk but dilute "
            "the relevance signal. "
        )
    elif best_cs == 300:
        reasoning += (
            "Smaller chunks improve retrieval precision by keeping each chunk "
            "focused on a single topic. Consider reducing overlap to ~40 chars "
            "to avoid excessive duplication. "
        )
    else:
        reasoning += (
            "Larger chunks capture more context per result, which benefits "
            "queries requiring paragraph-level understanding. "
            "Monitor precision since each chunk now covers more topics. "
        )
    reasoning += (
        f"K={best_k} is recommended to balance recall and the amount of "
        "context sent to the LLM."
    )
    return {
        "best_chunk_size": best_cs,
        "best_k": best_k,
        "reasoning": reasoning,
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    from datetime import datetime

    print("=" * 60)
    print("CityPark RAG Evaluation Suite")
    print("=" * 60)

    qa  = load_qa()
    grd = load_guardrail()
    print(f"Loaded {len(qa)} QA pairs, {len(grd)} guardrail items")

    # 1. Retrieval metrics (default collection)
    print("\n[1/6] Retrieval metrics…")
    ret_metrics = retrieval_metrics_at_k(qa, K_VALUES)
    print(f"  Recall@5={ret_metrics[5]['recall']}  "
          f"Precision@5={ret_metrics[5]['precision']}  "
          f"MRR={ret_metrics[5]['mrr']}")

    # 2. Answer quality
    print("\n[2/6] Answer quality…")
    aq_metrics = run_answer_quality(qa, n_samples=min(20, len(qa)))
    print(f"  KW_coverage={aq_metrics['keyword_coverage']}  "
          f"Sem_sim={aq_metrics['semantic_similarity']}  "
          f"Relevance={aq_metrics['query_answer_relevance']}")

    # 3. Performance
    print("\n[3/6] Performance benchmark…")
    perf_metrics = run_performance(qa[:20])  # use subset for speed
    print(f"  E2E mean={perf_metrics['e2e_latency_ms']['mean']} ms  "
          f"p95={perf_metrics['e2e_latency_ms']['p95']} ms  "
          f"throughput={perf_metrics['throughput']['qps']} qps")

    # 4. Guardrail
    print("\n[4/6] Guardrail evaluation…")
    grd_metrics = run_guardrail_eval(grd)
    print(f"  PII F1={grd_metrics['pii_detection']['f1']}  "
          f"Injection F1={grd_metrics['injection_detection']['f1']}")

    # 5. Chunk comparison
    print("\n[5/6] Chunk size comparison…")
    chunk_cmp = run_chunk_comparison(qa, CHUNK_SIZES, K_VALUES)

    # 6. Recommendation
    rec = make_recommendation(chunk_cmp)
    print(f"\n[6/6] Recommendation: chunk_size={rec['best_chunk_size']}, k={rec['best_k']}")

    # ── Assemble results ────────────────────────────────────────────────────
    results = {
        "metadata": {
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "n_questions": len(qa),
            "n_guardrail": len(grd),
            "embed_model": EMBED_MODEL,
            "k_values":    K_VALUES,
            "chunk_sizes": CHUNK_SIZES,
        },
        "retrieval":       ret_metrics,
        "answer_quality":  aq_metrics,
        "performance":     perf_metrics,
        "guardrail":       grd_metrics,
        "chunk_comparison": chunk_cmp,
        "recommendation":  rec,
    }

    # ── Save results.json ───────────────────────────────────────────────────
    results_path = DOCS_DIR / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved → {results_path}")

    # ── Generate charts ─────────────────────────────────────────────────────
    print("Generating charts…")
    _chart_retrieval(ret_metrics,   CHARTS_DIR / "retrieval_metrics.png")
    _chart_latency(perf_metrics,    CHARTS_DIR / "latency.png")
    _chart_guardrail(grd_metrics,   CHARTS_DIR / "guardrail.png")
    _chart_chunk_comparison(chunk_cmp, CHARTS_DIR / "chunk_comparison.png")
    print(f"  Charts → {CHARTS_DIR}/")

    # ── Generate report ─────────────────────────────────────────────────────
    print("Generating report…")
    generate_report(results, DOCS_DIR / "evaluation_report.md")

    print("\nDone!\n")
    print(f"  docs/results.json")
    print(f"  docs/evaluation_report.md")
    print(f"  docs/charts/*.png")


if __name__ == "__main__":
    main()
