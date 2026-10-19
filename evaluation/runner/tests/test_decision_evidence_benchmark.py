"""Grade provenance, binary semantics and reuse guards; never invokes models."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[3] / "deploy/docker/decision-eval"))
import decision_evidence_benchmark as e
import decision_capacity_preflight as capacity


def fixture(grades, cid="case"):
    candidates = [{"fragment_id": str(i), "content": "document " + str(i),
                   "content_sha256": hashlib.sha256(("document " + str(i)).encode()).hexdigest()}
                  for i in range(len(grades))]
    pool = {"case_id": cid, "query": "question " + cid,
            "query_identity_sha256": e.query_key("question " + cid), "candidates": candidates}
    label = {"case_id": cid, "grades": dict(zip((c["fragment_id"] for c in candidates), grades)),
             "positive_ids": [str(i) for i, g in enumerate(grades) if g >= 2]}
    return pool, label


def records(pool, chosen):
    return [{"case_id": pool["case_id"], "fragment_id": c["fragment_id"], "status": "valid",
             "prediction": "relevant" if i in chosen else "irrelevant"}
            for i, c in enumerate(pool["candidates"])]


class EvidenceTests(unittest.TestCase):
    def test_independent_original_source_join_and_corruption_checks(self):
        script = Path(__file__).parents[3] / "docs/research/2026-10-18-evidence-audit.py"
        spec = importlib.util.spec_from_file_location("evidence_offline_audit", script)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dataset = root / "source", root / "dataset"
            source.mkdir()
            dataset.mkdir()
            (source / "queries.dev.tsv").write_text("qid\ttext\n1\tquestion\n")
            (source / "qrels.dev.tsv").write_text("qid\t-\tpid\trel\n1\t0\ta\t1\n1\t0\tb\t2\n")
            (source / "collection.tsv").write_text("pid\ttext\na\ttopic only\nb\tanswer evidence\n")
            pp = {"case_id": "t2q:1", "source_qid": "1", "query": "question", "candidates": [
                {"fragment_id": "t2p:a", "source_pid": "a", "content": "topic only"},
                {"fragment_id": "t2p:b", "source_pid": "b", "content": "answer evidence"}]}
            ll = {"case_id": "t2q:1", "grades": {"t2p:a": 1, "t2p:b": 2}, "positive_ids": ["t2p:b"]}
            e.p.save(dataset / "protocol.json", {"split_qids": {"formal": ["1"]},
                "source_sha256": {f.name: e.sha(f) for f in source.iterdir()}})
            inputs, labels = dataset / "formal.inputs.private.jsonl", dataset / "formal.labels.private.jsonl"
            e.p.save_rows(inputs, [pp])
            e.p.save_rows(labels, [ll])
            self.assertTrue(audit.verify_original_sources(dataset, source)["passed"])
            bad = {**ll, "positive_ids": ["t2p:a", "t2p:b"]}
            labels.write_text(json.dumps(bad) + "\n")
            with self.assertRaises(AssertionError):
                audit.verify_original_sources(dataset, source)
            labels.write_text(json.dumps(ll) + "\n")
            pp["candidates"][1]["content"] = "shortened"
            inputs.write_text(json.dumps(pp) + "\n")
            with self.assertRaises(AssertionError):
                audit.verify_original_sources(dataset, source)

    def test_grades_are_not_old_binary_relevance(self):
        pool, label = fixture([0, 1, 2, 3])
        e.validate([pool], [label])
        self.assertEqual(label["positive_ids"], ["2", "3"])
        wrong = {**label, "positive_ids": ["1", "2", "3"]}
        with self.assertRaises(AssertionError):
            e.validate([pool], [wrong])

    def test_complete_content_hash_and_no_unjudged_candidates(self):
        pool, label = fixture([0, 2])
        for mutation in ("content", "unjudged", "grade"):
            pp, ll = deepcopy(pool), deepcopy(label)
            if mutation == "content":
                pp["candidates"][0]["content"] += " truncated"
            elif mutation == "unjudged":
                del ll["grades"]["0"]
            else:
                ll["grades"]["0"] = -1
            with self.assertRaises(AssertionError):
                e.validate([pp], [ll])

    def test_negative_only_pool_is_valid_and_scored_separately(self):
        pool, label = fixture([0, 1])
        e.validate([pool], [label])
        out = e.quality([pool], [label], records(pool, set()))
        self.assertEqual(out["correct_empty_rate"]["point"], 1)
        self.assertIsNone(out["positive_pool_macro"]["f1"])
        self.assertEqual(out["candidate_accuracy"], 1)
        out = e.quality([pool], [label], records(pool, {1}))
        self.assertEqual(out["correct_empty_rate"]["point"], 0)
        self.assertEqual(out["counts"]["fp"], 1)

    def test_empty_predictions_on_evidence_pool_are_misses(self):
        pool, label = fixture([0, 1, 2, 3])
        out = e.quality([pool], [label], records(pool, set()))
        self.assertEqual(out["positive_pool_macro"]["f1"]["point"], 0)
        self.assertEqual(out["missed_all_evidence_rate"]["point"], 1)
        self.assertEqual(out["counts"]["fn"], 2)
        self.assertIsNone(out["correct_empty_rate"])

    def test_all_keep_is_not_perfect_quality(self):
        pool, label = fixture([0, 1, 2, 3])
        out = e.quality([pool], [label], records(pool, {0, 1, 2, 3}))
        self.assertEqual(out["positive_pool_macro"]["precision"]["point"], .5)
        self.assertEqual(out["positive_pool_macro"]["recall"]["point"], 1)
        self.assertEqual(out["grades"]["1"]["accepted"], 1)
        self.assertEqual(out["character_reduction"], 0)

    def test_missing_or_failed_requests_do_not_receive_negative_credit(self):
        pool, label = fixture([0, 1, 2, 3])
        rows = records(pool, {2})[:3]
        rows[0]["status"] = "failed"
        out = e.quality([pool], [label], rows)
        self.assertEqual(out["counts"]["planned"], 4)
        self.assertEqual(out["counts"]["unscored"], 2)
        self.assertEqual(out["counts"].get("fn", 0), 0)
        self.assertEqual(out["candidate_accuracy"], 1)
        self.assertEqual(out["valid_pools"], 0)
        self.assertIsNone(out["positive_pool_macro"]["f1"])
        self.assertEqual(out["coverage_adjusted_zero_credit"]["candidate_accuracy"], .5)
        self.assertEqual(out["grades"]["3"]["unscored"], 1)
        self.assertIsNone(out["character_reduction"])

    def test_duplicate_or_extra_requests_rejected(self):
        pool, label = fixture([2])
        rows = records(pool, {0})
        with self.assertRaises(AssertionError):
            e.quality([pool], [label], rows + rows)
        with self.assertRaises(AssertionError):
            e.quality([pool], [label], [{**rows[0], "fragment_id": "unknown"}])

    def test_independent_deterministic_selection_includes_no_evidence(self):
        qs = {str(i): "query " + str(i) for i in range(30)}
        rs = {q: {"p": 0 if int(q) % 2 else 2} for q in qs}
        qs["29"] = qs["28"]
        selected, _ = e.select_queries(qs, rs, {e.query_key(qs["0"])},
                                      {"engineering": 2, "validation": 2, "formal": 5, "no_evidence": 3})
        keys = [e.query_key(qs[q]) for ids in selected.values() for q in ids]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertNotIn(e.query_key(qs["0"]), keys)
        self.assertTrue(all(rs[q]["p"] == 0 for q in selected["no_evidence"]))
        again, _ = e.select_queries(qs, rs, {e.query_key(qs["0"])},
                                   {"engineering": 2, "validation": 2, "formal": 5, "no_evidence": 3})
        self.assertEqual(selected, again)

    def test_modes_use_identical_rubric_without_noul_criteria(self):
        choice, noul = e.questions("choice"), e.questions("noul")
        self.assertEqual(choice["relevance"]["instructions"], noul["relevance"]["instructions"])
        self.assertNotIn("criteria", noul["relevance"])
        self.assertEqual(sum(len(e.modes(m)) for m in e.p.ROSTER), 28)

    def test_original_adapters_keep_default_contract(self):
        state = {"query": "q", "document": "d"}
        original = deepcopy(e.p.QUESTIONS)
        base = capacity.llm_body("Qwen3.8-27B", state)
        self.assertIn(original["relevance"]["instructions"], base["messages"][0]["content"])
        _, evidence = e.p.refine_wire("llm", "Qwen3.8-27B", state, e.questions("choice"))
        self.assertIn(e.INSTRUCTIONS, evidence["messages"][0]["content"])
        for key in ("temperature", "max_tokens", "response_format", "chat_template_kwargs"):
            self.assertEqual(evidence[key], base[key])
        self.assertEqual(e.p.QUESTIONS, original)
        _, native = e.p.refine_wire("worker", "model", state)
        self.assertEqual(native["questions"]["relevance"]["instructions"], original["relevance"]["instructions"])
        self.assertTrue(native["questions"]["relevance"]["require_full_input"])

    def test_decode_preserves_native_choice_even_if_probability_disagrees(self):
        raw = {"answers": {"relevance": {"type": "choice", "choice": "irrelevant",
               "probabilities": {"relevant": .8, "irrelevant": .2}}}}
        out = e.decode("sdk", "model", {}, raw, "choice")
        self.assertEqual(out["prediction"], "irrelevant")
        replay = e.decode("sdk", "model", {}, out["native_response"], "choice")
        for key in ("prediction", "probabilities", "native_response"):
            self.assertEqual(replay[key], out[key])

    def test_noul_fixed_readout_and_strict_llm(self):
        raw = {"answers": {"relevance": {"type": "noul", "noul": .5}}}
        self.assertEqual(e.decode("sdk", "model", {}, raw, "noul")["prediction"], "irrelevant")
        raw = {"model": "model", "choices": [{"finish_reason": "stop", "message": {"content": '{"choice":"relevant"}'}}]}
        out = e.decode("llm", "model", {}, raw, "choice")
        self.assertEqual(out["prediction"], "relevant")
        raw["choices"][0]["message"]["reasoning_content"] = "unexpected"
        with self.assertRaises(AssertionError):
            e.decode("llm", "model", {}, raw, "choice")

    def test_jev_choice_replay_preserves_one_model_call_and_no_thinking(self):
        raw = {"options": list(e.CRITERIA.values()), "probabilities": [.8, .2],
            "choice_index": 0, "choice": e.CRITERIA["relevant"], "num_model_requests": 1,
            "thinking": {"used": False}}
        parsed = e.decode("jev", "model", {}, raw, "choice")
        self.assertEqual(e.decode("jev", "model", {}, parsed["native_response"], "choice")["prediction"], "relevant")
        for changed in ({"num_model_requests": 2}, {"thinking": {"used": True}}):
            with self.assertRaises(AssertionError):
                e.decode("jev", "model", {}, {**raw, **changed}, "choice")

    def test_modes_alternate_by_pool_without_phase_overlap(self):
        pools = [fixture([0, 2], str(i))[0] for i in range(2)]
        seq = list(e.sequence("d3-mini", {"engineering": (pools, []), "formal": (pools, [])}))
        self.assertEqual([m for _, _, m in seq], ["choice.direct", "choice.top10", "noul.top10", "noul.direct",
            "choice.top10", "noul.direct", "choice.direct", "noul.top10"] * 2)
        self.assertEqual([phase for phase, _, _ in seq], ["engineering"] * 8 + ["formal"] * 8)
        self.assertEqual(sum(len(e.conditions(m)) for m in e.p.ROSTER), 56)

    def test_audit_checks_decodes_hashes_and_real_pool_walls(self):
        from launch_reviewed import MODELS
        model = "d3-edge"
        family, _, served = MODELS[model]
        pool, label = fixture([0] * 10 + [2, 3])
        data = {"formal": ([pool], [label])}
        import httpx
        sent = []
        def response(request):
            body = json.loads(request.content)
            sent.append(body)
            if request.url.path == "/rerank":
                self.assertEqual(body["documents"], [c["content"] for c in pool["candidates"]])
                raw = {"model": "Qwen3-Reranker-8B", "results": [
                    {"index": i, "relevance_score": .5} for i in range(12)]}
            else:
                mode = body["questions"]["relevance"]["type"]
                raw = {"answers": {"relevance": {"type": "choice", "choice": "relevant",
                    "probabilities": {"relevant": .8, "irrelevant": .2}} if mode == "choice" else {"type": "noul", "noul": .8}}}
            return httpx.Response(200, json=raw)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with httpx.Client(transport=httpx.MockTransport(response)) as http:
                transport = e.p.SerialHTTP(http, "secret", folder)
                try:
                    with (folder / "chains.jsonl").open("x") as stream:
                        for phase, pp, arm in e.sequence(model, data):
                            e.execute_chain(transport, model, family, served, phase, pp, arm, "http://reranker", stream)
                finally:
                    transport.close()
            rows = e.b.read_jsonl(folder / "requests.jsonl")
            chains = e.b.read_jsonl(folder / "chains.jsonl")
            for name in ("deployment", "container", "auth"):
                e.p.save(folder / (name + ".json"), {})
            result = e.audit_model(folder, model, data)
            self.assertEqual(result["calls"], 46)
            self.assertEqual(len(sent), 46)
            self.assertTrue(result["passed"])
            top = result["phases"]["formal"]["choice.top10"]
            self.assertEqual(top["quality"]["counts"]["fn"], 2)
            self.assertEqual(top["positive_omissions"]["reranker_pruned_positive"], 2)
            self.assertEqual(top["reranker_only"]["counts"]["planned"], 12)
            rr = next(r for r in rows if r["stage"] == "rerank")
            self.assertEqual(rr["retained_ids"], [str(i) for i in range(10)])
            self.assertEqual(len(e.p.rerank_result(pool, {"model": "Qwen3-Reranker-8B", "results": [
                {"index": i, "relevance_score": .5} for i in range(12)]})["retained_ids"]), 12)
            spec = importlib.util.spec_from_file_location("evidence_report", Path(__file__).parents[3] / "docs/research/2026-10-18-evidence-audit.py")
            report = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(report)
            delta = report.paired(e,
                result["phases"]["formal"]["choice.direct"]["quality"]["cases"],
                top["quality"]["cases"], [r for r in chains if r["arm"] == "choice.direct"],
                [r for r in chains if r["arm"] == "choice.top10"])
            original_quality = deepcopy(top["quality"])
            original_quality["coverage_adjusted_zero_credit"]["positive_pool_macro"]["f1"] = {
                "point": .1234, "low": .1, "high": .15}
            text = report.report_md({"models": {model: result}, "calls": 46, "passed": True,
                "baselines": {"formal": {}}, "common_coverage": {"covered": {"formal": {"pools": 1}}},
                "paired_comparisons": {"direct -> top10": {"formal": delta}},
                "original_denominator_zero_credit": {model: {"formal": {"choice.top10": original_quality}}},
                "online_reranker_variation": {"phases": {"formal": {"calls": 2}}, "policy": "per_request_top10_with_observed_variation"}})
            self.assertIn("choice.top10", text)
            self.assertIn("direct -> top10", text)
            self.assertIn("12.34 [10.00, 15.00]", text)
            self.assertIn("per_request_top10_with_observed_variation", text)
            self.assertLess(delta["f1"]["point"], 0)
            self.assertIn("各阶段实际覆盖与耗时", text)
            self.assertIn("裁剪与精筛漏判", text)
            bad_ranks = deepcopy(rows)
            bad_rank = next(r for r in bad_ranks if r["stage"] == "rerank")
            bad_rank["retained_ids"] = list(reversed(bad_rank["retained_ids"]))
            with patch.object(e.b, "read_jsonl", side_effect=[bad_ranks, rows, chains]):
                with self.assertRaises(AssertionError):
                    e.audit_model(folder, model, data)
            bad = deepcopy(rows)
            bad[0]["prediction"] = "irrelevant"
            with patch.object(e.b, "read_jsonl", side_effect=[bad, rows, chains]):
                with self.assertRaises(AssertionError):
                    e.audit_model(folder, model, data)
            bad_chains = deepcopy(chains)
            bad_chains[0]["wall_ms"] = 0
            with patch.object(e.b, "read_jsonl", side_effect=[rows, rows, bad_chains]):
                with self.assertRaises(AssertionError):
                    e.audit_model(folder, model, data)

    def test_timeout_chain_and_unattempted_ledger_no_retry(self):
        class HTTP:
            calls = 0
            def post(self, *args, **kwargs):
                self.calls += 1
                raise TimeoutError("private input must not be printed")
        pool, label = fixture([2, 0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "d3-edge"
            folder.mkdir()
            http = HTTP()
            transport = e.p.SerialHTTP(http, "secret", folder)
            try:
                with (folder / "chains.jsonl").open("x") as stream, self.assertRaises(e.p.HaltSequence):
                    e.execute_chain(transport, "d3-edge", "worker", "d3-edge", "formal", pool,
                        "choice.top10", "http://reranker", stream)
            finally:
                transport.close()
            e.failure_ledger(root, {"formal": ([pool], [label])}, "HaltSequence")
            ledger = e.b.read_jsonl(root / "failure-ledger.jsonl")
            failures = [r for r in ledger if r["http_calls"]]
            self.assertEqual(http.calls, 1)
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["reason_code"], "timeout")
            self.assertTrue(all(r["reason_code"] == "not_attempted_after_stop" for r in ledger if not r["http_calls"]))
            chain = e.b.read_jsonl(folder / "chains.jsonl")[0]
            self.assertFalse(chain["valid"])
            self.assertEqual(chain["calls"], 1)
            self.assertNotIn("private input", (folder / "requests.jsonl").read_text())

    def test_common_coverage_is_full_pool_and_all_native_modes(self):
        a, al = fixture([2, 0], "a")
        b, bl = fixture([0, 3], "b")
        data = {"formal": ([a, b], [al, bl])}
        rows = [{"phase": "formal", "case_id": pool["case_id"], "fragment_id": c["fragment_id"],
                 "input_sha256": e.b.digest(e.p.pair_state(pool, c)), "tokens": 10,
                 "effective_limit": 20, "supported": True} for pool in (a, b) for c in pool["candidates"]]
        conditions = [m + "." + mode for m in e.p.ROSTER for mode in e.modes(m)] + ["reranker"]
        coverage = {"dataset_protocol_sha256": "hash", "checks": {name: {"inference_calls": 0,
            "native_encoding": True, "encoding_source_sha256": {"encoder": "hash"}, "image_id": "image",
            "rows": deepcopy(rows)} for name in conditions}}
        coverage["checks"]["reranker"]["rows"][1].update(supported=False, tokens=21, reason="overlength")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage.json"
            path.write_text(json.dumps(coverage))
            active, excluded = e.load_coverage(path, data, "hash")
            self.assertEqual([p["case_id"] for p in active["formal"][0]], ["b"])
            self.assertEqual(excluded[("formal", "a")][0]["fragment_id"], "1")
            coverage["checks"].pop("reranker")
            path.write_text(json.dumps(coverage))
            with self.assertRaises(AssertionError):
                e.load_coverage(path, data, "hash")


class OnlineVariationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).parents[3] / "docs/research/2026-10-18-evidence-audit.py"
        spec = importlib.util.spec_from_file_location("online_variation_audit", script)
        cls.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.audit)

    def row(self, call_id, retained):
        return {"phase": "formal", "case_id": "pool", "model": "model", "arm": "choice.top10",
                "call_id": call_id, "input_sha256": "input", "request_sha256": "request",
                "scores": [{"fragment_id": str(i), "score": float(i)} for i in range(11)],
                "retained_ids": retained}

    def test_strict_default_still_rejects_order_or_membership_change(self):
        first = self.row("first", [str(i) for i in range(10)])
        for ids in (list(reversed(first["retained_ids"])), [str(i) for i in range(1, 11)]):
            with self.assertRaisesRegex(AssertionError, "Online Top10 membership changed"):
                self.audit.audit_online_rankings([first, self.row("second", ids)])

    def test_opt_in_classifies_against_first_not_previous_call(self):
        first = self.row("first", [str(i) for i in range(10)])
        order = self.row("order", list(reversed(first["retained_ids"])))
        member = self.row("member", [str(i) for i in range(1, 11)])
        restored = self.row("restored", first["retained_ids"])
        result = self.audit.audit_online_rankings([first, order, member, restored], True)
        self.assertEqual(result["counts"]["calls"], 4)
        self.assertEqual(result["counts"]["comparisons_to_first"], 3)
        self.assertEqual(result["counts"]["order_only"], 1)
        self.assertEqual(result["counts"]["membership_changed"], 1)
        self.assertEqual(result["counts"]["changed_pools"], 1)
        self.assertEqual(result["counts"]["same_ordered_top10"], 1)
        self.assertEqual(result["events"][1]["removed_ids"], ["0"])
        self.assertEqual(result["events"][1]["added_ids"], ["10"])
        self.assertEqual(result["events"][1]["baseline_call_id"], "first")

    def test_opt_in_never_allows_input_request_or_candidate_order_change(self):
        first = self.row("first", [str(i) for i in range(10)])
        for field in ("input_sha256", "request_sha256", "scores"):
            changed = deepcopy(first)
            changed[field] = list(reversed(first[field])) if field == "scores" else "changed"
            with self.assertRaises(AssertionError):
                self.audit.audit_online_rankings([first, changed], True)

    def test_scores_can_vary_without_changing_order_or_pool_identity(self):
        first = self.row("first", [str(i) for i in range(10)])
        changed = deepcopy(first)
        changed["call_id"] = "changed"
        changed["scores"][0]["score"] += .01
        other = self.row("other", [str(i) for i in range(1, 11)])
        other["phase"] = "engineering"
        result = self.audit.audit_online_rankings([first, changed, other])
        self.assertEqual(result["counts"]["score_changed_calls"], 1)
        self.assertEqual(result["counts"]["pools"], 2)
        self.assertEqual(result["events"], [])


if __name__ == "__main__":
    unittest.main()
