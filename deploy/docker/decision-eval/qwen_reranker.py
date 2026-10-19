"""Thin adapter for the official Qwen3 rerank prompt and yes/no logit scoring."""

from __future__ import annotations

import math
import time


DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def encode_records(tokenizer, body, max_length):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    instruction = body.get("instruction", DEFAULT_INSTRUCTION)
    texts = [f'<Instruct>: {instruction}\n<Query>: {body["query"]}\n<Document>: {doc}'
             for doc in body["documents"]]
    encoded = tokenizer(texts, padding=False, truncation=False, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
    if len(encoded) != len(texts):
        raise RuntimeError("Tokenizer returned incomplete documents")
    records = [{"input_ids": prefix + ids + suffix} for ids in encoded]
    if any(len(record["input_ids"]) > max_length for record in records):
        raise ValueError("Input exceeds the evaluation token limit")
    return records


class QwenReranker:
    def __init__(self, model_path, max_length):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left",
                                                       local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, trust_remote_code=False,
        ).to("cuda").eval()
        self.no_id, self.yes_id = [self.tokenizer.convert_tokens_to_ids(t) for t in ("no", "yes")]
        if (self.no_id == self.yes_id or self.tokenizer.encode("no", add_special_tokens=False) != [self.no_id]
                or self.tokenizer.encode("yes", add_special_tokens=False) != [self.yes_id]):
            raise RuntimeError("Qwen yes/no must each encode as one distinct token")

    def __call__(self, body):
        import torch

        if body.get("questions") is not None:
            raise ValueError("Qwen uses its official yes/no instruction, not decision questions")
        # Validate the entire pool first; an overlength later document must not
        # produce a partially successful pool or silently shortened input.
        records = encode_records(self.tokenizer, body, self.max_length)
        batch_size = body["batch_size"]
        device = self.model.device
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        scores, padded_tokens, forward_calls, gpu_ms = [], 0, 0, 0.0
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                chunk = records[start:start + batch_size]
                inputs = self.tokenizer.pad(chunk, padding=True, return_tensors="pt").to(device)
                padded_tokens += inputs["input_ids"].numel()
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                values = logits[:, [self.no_id, self.yes_id]].float().softmax(dim=-1)[:, 1]
                scores.extend(values.cpu().tolist())
                torch.cuda.synchronize(device)
                gpu_ms += (time.perf_counter() - started) * 1000
                forward_calls += 1
                del inputs, logits, values
        if len(scores) != len(records) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise RuntimeError("Reranker returned invalid or incomplete scores")
        lengths = [len(r["input_ids"]) for r in records]
        return {
            "model": body["model"],
            "results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)],
            "usage": {"input_tokens": sum(lengths), "output_tokens": 0},
            "execution": {
                "kind": "independent_gpu_batches", "batch_size": batch_size,
                "forward_calls": forward_calls, "input_tokens_per_document": lengths,
                "padded_tokens": padded_tokens, "gpu_inference_ms": round(gpu_ms, 3),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "dtype": str(self.model.dtype), "device": str(device),
                "backend": "transformers_sdpa", "truncated": False, "cpu_fallback": False,
            },
        }
