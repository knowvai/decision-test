"""Native tokenizer-only coverage adapters, run in pinned CPU-only images."""
import argparse
import asyncio
from dataclasses import replace
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def encoder(model, family, root, e):
    from transformers import AutoTokenizer
    sources = []
    if model.startswith("d3-"):
        from vllm_srun.families.decision3 import prompt
        tok, _ = prompt.load_tokenizer(root)
        prompt.check_chat_template(root)
        config = json.loads((root / "config.json").read_text())
        decision = json.loads((root / "decision_config.json").read_text())
        limit = decision.get("max_length") or config["text_config"]["max_position_embeddings"]
        sources.append(Path(inspect.getfile(prompt)))

        def encode(state, mode, body):
            q = body["questions"]["relevance"]
            text = prompt.render(prompt.user_prompt(state, q["instructions"],
                prompt.option_texts(mode, q.get("criteria")), decision["codes"][:2]))
            n = len(tok.encode(text, add_special_tokens=False).ids)
            return n, limit, n <= limit, {"reserved_output_tokens": 0}
    elif model.startswith("vela-"):
        from vllm_srun.families.vela2 import family as vf, members, package, calibration, request, dispatch
        cfg = json.loads((root / "config.json").read_text())
        cal = calibration.Calibration(json.loads((root / "calibration.json").read_text()))
        pkg = package.Vela2Package(root, cfg["model_type"], cfg, cal, (),
            root / "broad_head.safetensors" if (root / "broad_head.safetensors").exists() else None, {}, "", "")
        tok = vf._tokenizer(pkg)

        def planner(config):
            pp = replace(pkg, config=config)
            if pkg.member == package.ENCODER:
                member = members.EncoderMember.__new__(members.EncoderMember)
                member.layout = members.EncoderLayout(config)
            else:
                member = members.DecoderMember.__new__(members.DecoderMember)
                member.layout = members.DecoderLayout(config, dispatch.Dispatcher(cal, bool(pkg.broad_head)))
            return vf.Vela2Model(None, None, pp, member,
                request.QuestionReader(cal, bool(pkg.broad_head)), None, tok)

        actual, full = planner(cfg), planner({**cfg, "max_length": 1000000})
        sources += list(Path(inspect.getfile(vf)).parent.glob("*.py"))

        def encode(state, mode, body):
            unbounded = full.plan(state, body["questions"])
            assert not unbounded.errors and unbounded.items and all(m is not None for m in unbounded.mapping)
            planned = actual.plan(state, body["questions"])
            supported = not planned.errors and bool(planned.items) and all(m is not None for m in planned.mapping)
            n = max(len(item.ids) for item in unbounded.items)
            if supported:
                assert [list(x.ids) for x in planned.items] == [list(x.ids) for x in unbounded.items]
            return n, cfg["max_length"], supported, {"reserved_output_tokens": 0,
                "native_error_codes": [x.get("error") for x in planned.errors.values()]}
    elif family == "laya":
        from laya.agent import Agent
        from laya.common import build_sequence
        from server import validate_laya_input
        tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        agent = SimpleNamespace(tok=tok, _to_internal=Agent._to_internal)
        sources += [Path(inspect.getfile(x)) for x in (Agent, build_sequence, validate_laya_input)]

        def encode(state, mode, body):
            q = Agent._to_internal(body["questions"]["relevance"])
            ids, _, _ = build_sequence(tok, state, q, max_len=1000000, head_max_len=1000000, return_truncation_stats=True)
            try:
                counts = validate_laya_input(agent, body, 8192, 8192)
                assert counts == [len(ids)]
                supported = True
            except ValueError:
                supported = False
            return len(ids), 8192, supported, {"reserved_output_tokens": 0}
    elif model == "jeva-4b":
        from jeva.inference import Jeva, Choice, Noul
        from jeva._runtime.collate import tokenize_texts, state_question_text, option_text
        tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        obj = Jeva.__new__(Jeva)
        sources += [Path(inspect.getfile(x)) for x in (Jeva, tokenize_texts)]

        def encode(state, mode, body):
            q = body["questions"]["relevance"]
            item = Choice(instructions=q["instructions"], criteria=q["criteria"]) if mode == "choice" else Noul(instructions=q["instructions"])
            instruction, kind, _, options, *_ = obj._question(item)
            texts = [state_question_text(Jeva._state_text(state), instruction, kind)] + [option_text(x, kind) for x in options]
            _, lengths, truncations = tokenize_texts(tok, texts, 1000000, "error", field="coverage")
            assert not truncations
            n = max(lengths)
            if n <= 8192:
                _, actual, cuts = tokenize_texts(tok, texts, 8192, "error", field="coverage")
                assert actual == lengths and not cuts
            return n, 8192, n <= 8192, {"state_tokens": lengths[0], "option_tokens": lengths[1:], "reserved_output_tokens": 0}
    elif family == "sglang":
        from sglang.srt.entrypoints.systemone.joint_schema import encode_joint_schema, SystemOneRequest
        from sglang.srt.layers.joint_schema_head import max_joint_prompt_tokens
        tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
        # Native scheduler max_req_len is context-1; max_req_input_len is max_req_len-5.
        limit = max_joint_prompt_tokens(context_len=8192, num_reserved_tokens=0,
            max_req_input_len=8186, max_prefill_tokens=32768)
        sources += [Path(inspect.getfile(x)) for x in (encode_joint_schema, max_joint_prompt_tokens)]

        def encode(state, mode, body):
            request = SystemOneRequest(**body)
            ids, _ = encode_joint_schema(tok, request, max_length=1000000, image_token_counts=[])
            n = len(ids)
            if n <= limit:
                actual, _ = encode_joint_schema(tok, request, max_length=limit, image_token_counts=[])
                assert actual == ids
            return n, limit, n <= limit, {"reserved_output_tokens": 0, "native_scheduler_reserved_tokens": 7}
    elif family == "neohorse":
        sys.path.insert(0, str(root / "infer/vllm"))
        from head import CPUHead
        from jev_runtime.model import encode as jev_encode, rows_of
        from jev_runtime.schema import SystemOneRequest, to_record
        head = CPUHead.__new__(CPUHead)
        head.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        sources += [Path(inspect.getfile(x)) for x in (CPUHead, jev_encode, to_record)]

        def encode(state, mode, body):
            record, _ = to_record(SystemOneRequest(**body))
            packed = jev_encode(head.tokenizer, record, strict=True, option_isolation=False, max_state=1000000, max_branch=1000000)
            st, _, branches = rows_of(packed)
            sizes = [len(st) + len(branch["ids"]) for branch in branches]
            supported = len(st) <= 8192 and max(sizes) <= 12288 and sum(sizes) <= 32768
            if supported:
                assert [len(x["ids"]) for x in head.prepare(body)] == sizes
            return max(sizes), 12288, supported, {"state_tokens": len(st), "state_limit": 8192, "reserved_output_tokens": 0}
    elif family == "jev":
        spec = importlib.util.spec_from_file_location("native_jev", root / "serve_decide.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.setup(SimpleNamespace(lora_modules=[SimpleNamespace(name=module.LORA, path=str(root / "adapter_vllm"))],
            tokenizer=None, model=str(root), trust_remote_code=False, served_model_name="autotrust/JEV-27B", max_logprobs=20))
        tok = module.S["tok"]
        sources.append(root / "serve_decide.py")
        captured = []

        async def capture(ctx, content, has_img, ids, model, allowed=None):
            assert not has_img
            captured.append("".join(c["text"] for c in content))
            return {i: 0 for i in ids}

        module._logprobs = capture

        def encode(state, mode, body):
            parts, has_img = module._parts(body["state"])
            opts = body.get("options") or ["false", "true"]
            asyncio.run(module.s1_pass(None, mode, parts, has_img, body["question"], opts))
            assert len(captured) == 1 and module.READ["max_tokens"] == 1
            n = len(tok.encode(captured.pop(), add_special_tokens=module.READ["add_special_tokens"]))
            return n, 8191, n <= 8191, {"reserved_output_tokens": 1}
    elif family in ("llm", "reranker"):
        tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
        template = Path("/source/deploy/docker/decision-eval/qwen3_reranker.jinja")
        if family == "reranker":
            sources.append(template)

        def encode(state, mode, body):
            if family == "reranker":
                text = tok.apply_chat_template([{"role": "query", "content": state["query"]},
                    {"role": "document", "content": state["document"]}], chat_template=template.read_text(), tokenize=False)
                ids = tok.encode(text, add_special_tokens=True, truncation=False)
                reserve = 0
            else:
                ids = tok.apply_chat_template(body["messages"], tokenize=True, return_dict=False, add_generation_prompt=True,
                    **body.get("chat_template_kwargs", {}))
                reserve = body["max_tokens"]
            assert isinstance(ids, list) and ids and all(type(token) is int for token in ids)
            return len(ids), 8192 - reserve, len(ids) <= 8192 - reserve, {"reserved_output_tokens": reserve}
    else:
        raise ValueError("Unsupported native encoding family")
    sources += [path for path in root.rglob("*") if path.name in
        ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", "decision_config.json", "chat_template.jinja", "calibration.json")]
    return encode, sources


def main(args):
    sys.path.insert(0, "/source/evaluation/runner/scripts")
    sys.path.insert(0, "/source/deploy/docker/decision-eval")
    import decision_evidence_benchmark as e
    from launch_reviewed import MODELS
    family, _, served = MODELS[args.model]
    encode, sources = encoder(args.model, family, Path("/model"), e)
    data, _ = e.load_dataset(Path("/dataset"))
    checks = {}
    for mode in ("reranker",) if family == "reranker" else e.modes(args.model):
        rows = []
        for phase, (pools, _) in data.items():
            for pool in pools:
                for candidate in pool["candidates"]:
                    state = e.p.pair_state(pool, candidate)
                    body = {} if family == "reranker" else e.p.refine_wire(family, served, state, e.questions(mode))[1]
                    n, limit, supported, extra = encode(state, mode, body)
                    rows.append({"phase": phase, "case_id": pool["case_id"], "fragment_id": candidate["fragment_id"],
                        "input_sha256": e.b.digest(state), "tokens": int(n), "effective_limit": int(limit),
                        "supported": bool(supported), "reason": None if supported else "native_complete_input_exceeds_budget", **extra})
        condition = "reranker" if family == "reranker" else args.model + "." + mode
        checks[condition] = {"inference_calls": 0, "native_encoding": True, "image_id": args.image_id,
            "encoding_source_sha256": {str(p): e.sha(p) for p in set(sources)}, "rows": rows}
    with args.output.open("x") as stream:
        json.dump({"dataset_protocol_sha256": e.sha("/dataset/protocol.json"), "checks": checks}, stream)
        stream.write("\n")
    print(json.dumps({k: {"pairs": len(v["rows"]), "unsupported": sum(not r["supported"] for r in v["rows"]),
        "max_tokens": max(r["tokens"] for r in v["rows"])} for k, v in checks.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
