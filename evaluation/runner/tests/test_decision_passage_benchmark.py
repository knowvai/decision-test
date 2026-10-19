"""Passage-only invariants; no network, GPU, product imports or real inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).parents[3] / "deploy/docker/decision-eval"))
import decision_passage_benchmark as passage


def pool(count=3, case="pool"):
    return {"case_id": case, "task": "reranking", "query": "fixture query",
            "candidates": [{"fragment_id": hashlib.sha256(f"passage {i}".encode()).hexdigest(),
                            "content": f"passage {i}"} for i in range(count)]}


def labels(p):
    return {"case_id": p["case_id"], "task": "reranking", "positive_ids": [p["candidates"][0]["fragment_id"]]}


class DatasetTests(unittest.TestCase):
    def test_mask_only_explicit_fragments_without_mutating_sources(self):
        p = pool()
        original = json.dumps(p)
        exclude = {"case_id": p["case_id"], "fragment_id": p["candidates"][1]["fragment_id"]}
        revised, gold = passage.mask_fragments([p], [labels(p)], [exclude])
        self.assertEqual(len(revised), 1)
        self.assertEqual([c["content"] for c in revised[0]["candidates"]], ["passage 0", "passage 2"])
        self.assertEqual(gold, [labels(p)])
        self.assertEqual(json.dumps(p), original)

    def test_mask_updates_gold_and_rejects_duplicates_unknown_and_empty_gold(self):
        p = pool()
        gold = {**labels(p), "positive_ids": [c["fragment_id"] for c in p["candidates"][:2]]}
        ex = {"case_id": p["case_id"], "fragment_id": p["candidates"][0]["fragment_id"]}
        _, revised = passage.mask_fragments([p], [gold], [ex])
        self.assertEqual(revised[0]["positive_ids"], [p["candidates"][1]["fragment_id"]])
        for exclusions in ([ex, ex], [{**ex, "case_id": "unknown"}]):
            with self.assertRaises(AssertionError):
                passage.mask_fragments([p], [gold], exclusions)
        with self.assertRaises(AssertionError):
            passage.mask_fragments([p], [labels(p)], [ex])

    def test_requires_content_hash_and_unique_coverage(self):
        p = pool()
        p["candidates"][1]["content"] = "different"
        with self.assertRaises(AssertionError):
            passage.validate_pools([p])


class SelectionTests(unittest.TestCase):
    def test_top29_stable_original_order_and_complete_pool(self):
        p = pool(35)
        raw = {"model": "Qwen3-Reranker-8B", "results": [
            {"index": i, "relevance_score": .5} for i in reversed(range(35))]}
        result = passage.rerank_result(p, raw)
        self.assertEqual(result["retained_ids"], [c["fragment_id"] for c in p["candidates"][:29]])
        self.assertEqual(len(result["scores"]), 35)
        for results in (raw["results"][:-1], raw["results"] + [raw["results"][0]],
                        [{"index": i, "relevance_score": float("nan")} for i in range(35)]):
            with self.assertRaises(AssertionError):
                passage.rerank_result(p, {**raw, "results": results})

    def test_small_pool_retains_all_and_scores_do_not_use_gold(self):
        p = pool()
        raw = {"model": "Qwen3-Reranker-8B", "results": [
            {"index": i, "relevance_score": i / 10} for i in range(3)]}
        self.assertEqual(set(passage.rerank_result(p, raw)["retained_ids"]),
                         {c["fragment_id"] for c in p["candidates"]})
        self.assertEqual(passage.set_metrics({"a", "b"}, {"a"}), {"precision": 1, "recall": .5, "f1": 2/3})
        self.assertEqual(passage.set_metrics({"a"}, set()), {"precision": 0, "recall": 0, "f1": 0})
        self.assertEqual(passage.set_metrics({"a"}, {"a"}, False)["f1"], 0)

    def test_threshold_tie_uses_highest_and_exact_inclusive_boundary(self):
        p = pool(2)
        rows = [{"case_id": p["case_id"], "fragment_id": c["fragment_id"], "status": "valid",
                 "probabilities": {"relevant": score, "irrelevant": 1-score}} for c,score in zip(p["candidates"], [.8, .2])]
        rule = passage.calibrate([p], [labels(p)], rows, True)
        self.assertEqual(rule["threshold"], .8)
        self.assertEqual(passage.selected(rows, rule), set(labels(p)["positive_ids"]))
        rows[0]["status"] = "failed"
        rule = passage.calibrate([p], [labels(p)], rows, True)
        self.assertEqual(rule["macro_f1"], 0)
        self.assertGreater(rule["threshold"], 1)

    def test_hard_decisions_never_fabricate_scores(self):
        p = pool(2)
        rows = [{"case_id": p["case_id"], "fragment_id": c["fragment_id"], "status": "valid",
                 "prediction": "relevant" if i == 0 else "irrelevant"} for i,c in enumerate(p["candidates"])]
        rule = passage.calibrate([p], [labels(p)], rows, False)
        self.assertEqual(rule["kind"], "hard_choice")
        self.assertNotIn("threshold", rule)
        self.assertEqual(rule["macro_f1"], 1)

    def test_native_and_llm_wire_are_reused_without_other_candidates(self):
        p = pool()
        state = passage.pair_state(p, p["candidates"][0])
        path, body = passage.refine_wire("llm", "Qwen3.8-27B", state)
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(json.loads(body["messages"][1]["content"]), state)
        _, body = passage.refine_wire("worker", "Vela-2.0-4B", state)
        self.assertTrue(body["questions"]["relevance"]["require_full_input"])


class TransportTests(unittest.TestCase):
    def test_journals_single_call_and_refuses_replay(self):
        class HTTP:
            calls = 0
            def post(self, *args, **kwargs):
                self.calls += 1
                class Response:
                    status_code, content = 200, b'{}'
                    def raise_for_status(self): pass
                    def json(self): return {}
                return Response()
        with tempfile.TemporaryDirectory() as directory:
            http = HTTP()
            client = passage.SerialHTTP(http, "secret", Path(directory))
            identity = {"case_id": "test"}
            row = client.call(identity, "http://127.0.0.1:1", lambda: ("/path", {"a": 1}), lambda raw: {"prediction": "relevant"}, "hash")
            self.assertEqual(row["calls"], 1)
            self.assertEqual(row["status"], "valid")
            self.assertEqual(client.active, 0)
            with self.assertRaises(AssertionError):
                client.call(identity, "", lambda: ("/path", {}), lambda raw: {}, "hash")
            client.close()
            self.assertEqual(http.calls, 1)
            for filename in ("intents.jsonl", "requests.jsonl"):
                text = (Path(directory) / filename).read_text()
                self.assertNotIn("secret", text)
                self.assertEqual(len(text.splitlines()), 1)

    def test_timeout_stops_after_preserving_failure(self):
        class HTTP:
            def post(self, *args, **kwargs): raise TimeoutError("sensitive body must not appear")
        with tempfile.TemporaryDirectory() as directory:
            client = passage.SerialHTTP(HTTP(), "secret", Path(directory))
            with self.assertRaises(passage.HaltSequence):
                client.call({"case_id": "test"}, "", lambda: ("/path", {}), lambda raw: {}, "hash")
            with self.assertRaises(passage.HaltSequence):
                client.call({"case_id": "must-not-send"}, "", lambda: ("/path", {}), lambda raw: {}, "hash")
            client.close()
            text = (Path(directory) / "requests.jsonl").read_text()
            row = json.loads(text)
            self.assertEqual(row["calls"], 1)
            self.assertEqual(row["status"], "failed")
            self.assertTrue(row["sequence_halted"])
            self.assertFalse(row["server_terminal_confirmed"])
            self.assertNotIn("sensitive", text)

    def test_400_records_failure_without_automatic_retry(self):
        class HTTP:
            def post(self, *args, **kwargs):
                class Response:
                    status_code, content = 400, b'{}'
                    def raise_for_status(self): raise ValueError("bad request")
                return Response()
        with tempfile.TemporaryDirectory() as directory:
            client = passage.SerialHTTP(HTTP(), "secret", Path(directory))
            row = client.call({"case_id": "test"}, "", lambda: ("/path", {}), lambda raw: {}, "hash")
            client.close()
            self.assertEqual(row["status"], "failed")
            self.assertFalse(row["sequence_halted"])
            self.assertEqual(row["calls"], 1)

    def test_output_refuses_existing_attempt_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            client = passage.SerialHTTP(None, "secret", Path(directory))
            client.close()
            with self.assertRaises(FileExistsError):
                passage.SerialHTTP(None, "secret", Path(directory))


class MetricsTests(unittest.TestCase):
    def test_failed_pool_zero_denominator_and_paired_timing_same_intersection(self):
        pools = [pool(case="a"), pool(case="b")]
        chains = [{"arm": arm, "case_id": p["case_id"], "status": "failed" if arm == "B3" and p["case_id"] == "b" else "valid",
                   "selected_ids": labels(p)["positive_ids"], "wall_ms": 100 if arm == "B1" else 80}
                  for p in pools for arm in ("B1", "B3")]
        summary = passage.score_chains(pools, [labels(p) for p in pools], chains)
        self.assertEqual(summary["arms"]["B3"]["quality"]["f1"]["point"], .5)
        self.assertEqual(summary["paired"]["timed_successful_pairs"], 1)
        self.assertEqual(summary["paired"]["ratio_of_summed_paired_times"], 1.25)
        self.assertEqual(summary["paired"]["quality_delta"]["f1"]["point"], -.5)
        self.assertEqual(passage.request_stats([])["successful_latency_ms"]["mean_ms"], None)


class OnlineFlowTests(unittest.TestCase):
    def test_complete_serial_chain_uses_full_rerank_pool_then_top29(self):
        class Response:
            status_code = 200
            def __init__(self, raw):
                self.raw = raw
                self.content = json.dumps(raw).encode()
            def json(self): return self.raw
            def raise_for_status(self): pass
        class HTTP:
            def __init__(self): self.sent = []
            def post(self, endpoint, json, **kwargs):
                self.sent.append((endpoint, json))
                if endpoint.endswith("/rerank"):
                    return Response({"model": "Qwen3-Reranker-8B", "results": [
                        {"index": i, "relevance_score": 1 - i / 100} for i in range(len(json["documents"]))]})
                score = .8 if json["state"]["document"] == "passage 0" else .2
                return Response({"answers": {"relevance": {"type": "choice",
                    "choice": "relevant" if score > .5 else "irrelevant",
                    "probabilities": {"relevant": score, "irrelevant": 1-score}}}})
        engineering, validation = [pool(2, "engineering")], [pool(2, "validation")]
        formal = [pool(35, "formal-a"), pool(3, "formal-b")]
        dataset = {"engineering": engineering, "validation": validation,
                   "validation_labels": [labels(p) for p in validation],
                   "formal": formal, "formal_labels": [labels(p) for p in formal]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "model"
            folder.mkdir()
            args = SimpleNamespace(timeout=300, startup_timeout=2400, output=root,
                                   base_url="http://127.0.0.1:1", reranker_url="http://192.168.0.2:2")
            http, reference, statuses = HTTP(), {}, []
            with patch.object(passage, "readiness", return_value={"ready": 200}), \
                 patch.object(passage, "container_identity", return_value={"id": "stable"}):
                passage.execute_model(args, "d3-edge", folder, http, "secret", dataset, reference,
                                      lambda stage, **values: statuses.append(stage))
            chains = passage.benchmark.read_jsonl(folder / "chains.jsonl")
            requests = passage.benchmark.read_jsonl(folder / "requests.jsonl")
            self.assertEqual([c["arm"] for c in chains], ["B1", "B3", "B3", "B1"])
            self.assertEqual([len(b["documents"]) for url,b in http.sent if url.endswith("/rerank")], [35, 3])
            self.assertEqual([c["attempted_refine"] for c in chains], [35, 29, 3, 3])
            self.assertEqual([c["pruned"] for c in chains], [0, 6, 0, 0])
            self.assertTrue(all(c["status"] == "valid" for c in chains))
            self.assertEqual(len({r["call_id"] for r in requests}), len(requests))
            self.assertTrue(all(r["calls"] == r["inflight_at_start"] == 1 for r in requests))
            self.assertLessEqual(max(r["ended_seconds"] for r in requests[:-1]), requests[-1]["started_seconds"])
            self.assertEqual(len(reference["formal-a"]), 29)
            self.assertEqual(statuses[:2], ["engineering", "calibrating"])
            self.assertEqual(json.loads((folder / "threshold.json").read_text())["threshold"], .8)
            audit = passage.audit_model(folder, dataset, "d3-edge")
            self.assertTrue(audit["request_hashes_reconstructed"])
            self.assertTrue(audit["metrics_recomputed"])

    def test_engineering_failure_blocks_calibration_and_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(timeout=300, startup_timeout=2400, output=root,
                                   base_url="http://127.0.0.1:1", reranker_url="http://192.168.0.2:2")
            statuses = []
            with patch.object(passage, "readiness", return_value={"ready": 200}), \
                 patch.object(passage, "container_identity", return_value={"id": "stable"}), \
                 patch.object(passage.SerialHTTP, "call", return_value={"status": "failed"}), \
                 self.assertRaises(AssertionError):
                passage.execute_model(args, "d3-edge", root, None, "secret", {"engineering": [pool()]}, {},
                                      lambda stage, **values: statuses.append(stage))
            self.assertEqual(statuses, ["engineering"])
            self.assertFalse((root / "threshold.json").exists())
            self.assertFalse((root / "chains.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
