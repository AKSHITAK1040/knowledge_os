from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import time
from typing import Any
import urllib.request
import urllib.error


def send_request(base_url: str, api_key: str, query: str) -> dict[str, Any]:
    url = f"{base_url}/v1/chat"
    payload = json.dumps({"query": query, "top_k": 5}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "success": True,
                "status_code": resp.status,
                "latency_ms": elapsed_ms,
                "tokens": len(data.get("answer", "").split()),
                "confidence": data.get("confidence", 0.0),
            }
    except urllib.error.HTTPError as err:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"success": False, "status_code": err.code, "latency_ms": elapsed_ms, "error": str(err)}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"success": False, "status_code": 500, "latency_ms": elapsed_ms, "error": str(exc)}


def run_load_test(
    base_url: str = "http://127.0.0.1:8000",
    api_key: str = "dev-key",
    total_requests: int = 20,
    concurrency: int = 5,
) -> dict[str, Any]:
    queries = [
        "What are the benefits of multi-agent supervisor architectures?",
        "How does Reciprocal Rank Fusion improve hybrid retrieval?",
        "Explain RAG Triad evaluation and hallucination detection.",
        "What is the role of episodic memory in long-running AI research?",
        "Compare dense vector search against BM25Okapi keyword retrieval.",
    ]

    print(f"🚀 Starting KnowledgeOS Load Test: {total_requests} requests with concurrency {concurrency}...")
    start_total = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, base_url, api_key, queries[i % len(queries)])
            for i in range(total_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    total_time_s = time.perf_counter() - start_total
    successful = [r for r in results if r["success"]]
    latencies = [r["latency_ms"] for r in successful]

    rps = len(results) / max(0.001, total_time_s)
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0
    p99 = sorted(latencies)[max(0, int(len(latencies) * 0.99) - 1)] if latencies else 0.0

    report = {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "success_rate_percent": round((len(successful) / max(1, len(results))) * 100, 2),
        "total_time_seconds": round(total_time_s, 2),
        "throughput_rps": round(rps, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "min_latency_ms": round(min(latencies), 2) if latencies else 0.0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0.0,
    }

    print("\n📊 Load Test Results:")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_load_test()
