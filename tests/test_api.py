from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from knowledgeos.api.app import app


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.headers = {"x-api-key": "dev-key", "x-user-id": "test-user", "x-role": "admin"}

    def test_health_endpoint(self) -> None:
        res = self.client.get("/health", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("stats", data)

        # Unauthenticated probe (container health check, Kubernetes liveness probe)
        unauth_res = self.client.get("/health")
        self.assertEqual(unauth_res.status_code, 200)
        self.assertEqual(unauth_res.json()["status"], "ok")

    def test_chat_endpoint(self) -> None:
        res = self.client.post(
            "/v1/chat",
            json={"query": "Explain supervisor orchestration", "top_k": 3},
            headers=self.headers,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("answer", data)
        self.assertIn("confidence", data)
        self.assertIn("execution_ms", data)

    def test_ingest_and_chunks_endpoint(self) -> None:
        ingest_res = self.client.post(
            "/v1/ingest",
            json={"text": "KnowledgeOS provides state of the art multi-agent research.", "title": "API Test Doc"},
            headers=self.headers,
        )
        self.assertEqual(ingest_res.status_code, 200)
        self.assertGreater(ingest_res.json()["chunks_created"], 0)

        chunks_res = self.client.get("/v1/chunks", headers=self.headers)
        self.assertEqual(chunks_res.status_code, 200)
        self.assertGreater(chunks_res.json()["total_persisted"], 0)

    def test_memory_crud_endpoints(self) -> None:
        # Create memory
        create_res = self.client.post(
            "/v1/memory",
            json={"user_id": "test-user", "session_id": "s1", "summary": "User prefers concise summaries", "importance": 0.8},
            headers=self.headers,
        )
        self.assertEqual(create_res.status_code, 200)
        mem_id = create_res.json()["id"]

        # Get memory
        get_res = self.client.get("/v1/memory?user_id=test-user", headers=self.headers)
        self.assertEqual(get_res.status_code, 200)
        self.assertTrue(any(m["id"] == mem_id for m in get_res.json()["user"]))

        # Delete memory
        del_res = self.client.delete(f"/v1/memory/{mem_id}", headers=self.headers)
        self.assertEqual(del_res.status_code, 200)
        self.assertTrue(del_res.json()["deleted"])

    def test_analytics_and_traces_endpoints(self) -> None:
        analytics_res = self.client.get("/v1/analytics", headers=self.headers)
        self.assertEqual(analytics_res.status_code, 200)
        self.assertIn("average_latency_ms", analytics_res.json())

        traces_res = self.client.get("/v1/traces", headers=self.headers)
        self.assertEqual(traces_res.status_code, 200)
        self.assertIn("items", traces_res.json())

    def test_auth_rejection(self) -> None:
        # Missing auth header should be rejected on protected endpoints
        res = self.client.get("/v1/info")
        self.assertEqual(res.status_code, 401)

        # Invalid key should be rejected
        res_bad = self.client.get("/v1/info", headers={"x-api-key": "wrong-key"})
        self.assertEqual(res_bad.status_code, 401)

    def test_system_info_endpoint(self) -> None:
        res = self.client.get("/v1/info", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["app_name"], "KnowledgeOS")
        self.assertIn("stats", data)

    def test_memory_compress_endpoint(self) -> None:
        # Add memories
        self.client.post(
            "/v1/memory",
            json={"user_id": "compress-user", "session_id": "s1", "summary": "quantum theory overview", "importance": 0.5},
            headers=self.headers,
        )
        self.client.post(
            "/v1/memory",
            json={"user_id": "compress-user", "session_id": "s2", "summary": "quantum theory overview notes", "importance": 0.6},
            headers=self.headers,
        )
        res = self.client.post("/v1/memory/compress?user_id=compress-user", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        self.assertIn("compressed_items", res.json())

    def test_prometheus_metrics_endpoint(self) -> None:
        res = self.client.get("/metrics")
        self.assertEqual(res.status_code, 200)
        self.assertIn("knowledgeos_requests_total", res.text)


if __name__ == "__main__":
    unittest.main()
