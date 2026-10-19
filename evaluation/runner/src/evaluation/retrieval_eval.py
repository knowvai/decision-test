"""产品 证据级检索评测。

这层只评测已经收集好的检索结果，不直接依赖线上服务。这样可以低成本、可重复地比较不同检索策略。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_STAGE = "final"


@dataclass(frozen=True)
class Evidence:
    fragment_id: Optional[str] = None
    quote: Optional[str] = None
    importance: str = "supporting"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Evidence":
        return cls(
            fragment_id=_clean_optional_str(data.get("fragment_id")),
            quote=_clean_optional_str(data.get("quote")),
            importance=str(data.get("importance") or "supporting"),
        )

    def key(self) -> str:
        if self.fragment_id:
            return f"fragment:{self.fragment_id}"
        if self.quote:
            return f"quote:{self.quote}"
        return "empty"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "quote": self.quote,
            "importance": self.importance,
        }


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    kb_ids: List[str] = field(default_factory=list)
    expected_evidence: List[Evidence] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    forbidden_kb_ids: List[str] = field(default_factory=list)
    strict_kb_scope: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalCase":
        return cls(
            case_id=str(data["case_id"]),
            query=str(data.get("query") or ""),
            kb_ids=[str(item) for item in data.get("kb_ids", []) if str(item).strip()],
            expected_evidence=[
                Evidence.from_dict(item)
                for item in data.get("expected_evidence", [])
                if isinstance(item, Mapping)
            ],
            tags=[str(item) for item in data.get("tags", []) if str(item).strip()],
            forbidden_kb_ids=[
                str(item) for item in data.get("forbidden_kb_ids", []) if str(item).strip()
            ],
            strict_kb_scope=bool(data.get("strict_kb_scope", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "kb_ids": self.kb_ids,
            "expected_evidence": [item.to_dict() for item in self.expected_evidence],
            "tags": self.tags,
            "forbidden_kb_ids": self.forbidden_kb_ids,
            "strict_kb_scope": self.strict_kb_scope,
        }


@dataclass(frozen=True)
class RetrievedItem:
    fragment_id: Optional[str] = None
    content: str = ""
    kb_id: Optional[str] = None
    stage: str = DEFAULT_STAGE
    rank: int = 0
    score: Optional[float] = None
    source_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    order: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], order: int) -> "RetrievedItem":
        rank = data.get("rank")
        return cls(
            fragment_id=_clean_optional_str(data.get("fragment_id")),
            content=str(data.get("content") or ""),
            kb_id=_clean_optional_str(data.get("kb_id") or data.get("knowledge_base_id")),
            stage=str(data.get("stage") or DEFAULT_STAGE),
            rank=int(rank) if isinstance(rank, int) or str(rank).isdigit() else 0,
            score=_clean_optional_float(data.get("score")),
            source_type=_clean_optional_str(data.get("source_type")),
            metadata=dict(data.get("metadata") or {}),
            order=order,
        )

    def sort_key(self) -> tuple[int, int]:
        rank = self.rank if self.rank > 0 else self.order + 1
        return rank, self.order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "content": self.content,
            "kb_id": self.kb_id,
            "stage": self.stage,
            "rank": self.rank,
            "score": self.score,
            "source_type": self.source_type,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EvalRunResult:
    case_id: str
    retrieved_items: List[RetrievedItem] = field(default_factory=list)
    timing_ms: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalRunResult":
        raw_items = data.get("retrieved_items", [])
        if not raw_items and "records" in data:
            raw_items = data.get("records", [])

        return cls(
            case_id=str(data["case_id"]),
            retrieved_items=[
                RetrievedItem.from_dict(item, order=index)
                for index, item in enumerate(raw_items)
                if isinstance(item, Mapping)
            ],
            timing_ms={str(key): int(value) for key, value in dict(data.get("timing_ms") or {}).items()},
        )


@dataclass(frozen=True)
class IngestedFragment:
    fragment_id: Optional[str] = None
    content: str = ""
    kb_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IngestedFragment":
        metadata = dict(data.get("metadata") or {})
        return cls(
            fragment_id=_clean_optional_str(data.get("fragment_id") or data.get("id")),
            content=str(data.get("content") or data.get("page_content") or ""),
            kb_id=_clean_optional_str(data.get("kb_id") or data.get("knowledge_base_id")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class StageMetrics:
    case_count: int
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_count": self.case_count,
            "hit_at_k": _string_keyed_float_dict(self.hit_at_k),
            "recall_at_k": _string_keyed_float_dict(self.recall_at_k),
            "precision_at_k": _string_keyed_float_dict(self.precision_at_k),
            "ndcg_at_k": _string_keyed_float_dict(self.ndcg_at_k),
            "mrr": round(self.mrr, 6),
        }


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    tags: List[str]
    stage_metrics: Dict[str, StageMetrics]
    permission_leaks: List[RetrievedItem]
    ingestion_gaps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tags": self.tags,
            "stage_metrics": {stage: metrics.to_dict() for stage, metrics in self.stage_metrics.items()},
            "permission_leaks": [item.to_dict() for item in self.permission_leaks],
            "ingestion_gaps": self.ingestion_gaps,
        }


@dataclass(frozen=True)
class EvalSummary:
    total_cases: int
    evaluated_stages: List[str]
    permission_leak_cases: int
    permission_leak_rate: float
    ingestion_gap_cases: int = 0
    ingestion_gap_rate: float = 0.0
    ingestion_complete_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "evaluated_stages": self.evaluated_stages,
            "permission_leak_cases": self.permission_leak_cases,
            "permission_leak_rate": round(self.permission_leak_rate, 6),
            "ingestion_gap_cases": self.ingestion_gap_cases,
            "ingestion_gap_rate": round(self.ingestion_gap_rate, 6),
            "ingestion_complete_rate": round(self.ingestion_complete_rate, 6),
        }


@dataclass(frozen=True)
class TagReport:
    tag: str
    case_count: int
    stage_metrics: Dict[str, StageMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "case_count": self.case_count,
            "stage_metrics": {stage: metrics.to_dict() for stage, metrics in self.stage_metrics.items()},
        }


@dataclass(frozen=True)
class EvalReport:
    summary: EvalSummary
    stage_metrics: Dict[str, StageMetrics]
    by_tag: Dict[str, TagReport]
    case_results: List[CaseResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "stage_metrics": {stage: metrics.to_dict() for stage, metrics in self.stage_metrics.items()},
            "by_tag": {tag: report.to_dict() for tag, report in self.by_tag.items()},
            "case_results": [item.to_dict() for item in self.case_results],
        }


def load_cases_jsonl(path: str | Path) -> List[EvalCase]:
    return [EvalCase.from_dict(item) for item in _read_jsonl(path)]


def load_results_jsonl(path: str | Path) -> Dict[str, EvalRunResult]:
    results = [EvalRunResult.from_dict(item) for item in _read_jsonl(path)]
    return {result.case_id: result for result in results}


def load_ingestion_manifest_jsonl(path: str | Path) -> List[IngestedFragment]:
    return [IngestedFragment.from_dict(item) for item in _read_jsonl(path)]


def evaluate_cases(
    cases: Sequence[EvalCase],
    results: Mapping[str, EvalRunResult],
    k_values: Sequence[int] = (1, 3, 5, 10),
    ingestion_manifest: Optional[Sequence[IngestedFragment]] = None,
) -> EvalReport:
    normalized_k_values = sorted({int(k) for k in k_values if int(k) > 0})
    stages = _collect_stages(results)
    ingestion_index = _build_ingestion_index(ingestion_manifest)
    case_results = [
        _evaluate_case(
            case,
            results.get(case.case_id),
            stages,
            normalized_k_values,
            ingestion_index,
        )
        for case in cases
    ]

    stage_metrics = {
        stage: _aggregate_stage_metrics(
            [case_result.stage_metrics[stage] for case_result in case_results],
            normalized_k_values,
        )
        for stage in stages
    }

    tags = sorted({tag for case_result in case_results for tag in case_result.tags})
    by_tag = {
        tag: TagReport(
            tag=tag,
            case_count=len([case_result for case_result in case_results if tag in case_result.tags]),
            stage_metrics={
                stage: _aggregate_stage_metrics(
                    [
                        case_result.stage_metrics[stage]
                        for case_result in case_results
                        if tag in case_result.tags
                    ],
                    normalized_k_values,
                )
                for stage in stages
            },
        )
        for tag in tags
    }

    permission_leak_cases = sum(1 for case_result in case_results if case_result.permission_leaks)
    ingestion_gap_cases = sum(1 for case_result in case_results if case_result.ingestion_gaps)
    total_cases = len(case_results)
    summary = EvalSummary(
        total_cases=total_cases,
        evaluated_stages=stages,
        permission_leak_cases=permission_leak_cases,
        permission_leak_rate=permission_leak_cases / total_cases if total_cases else 0.0,
        ingestion_gap_cases=ingestion_gap_cases,
        ingestion_gap_rate=ingestion_gap_cases / total_cases if total_cases else 0.0,
        ingestion_complete_rate=(total_cases - ingestion_gap_cases) / total_cases if total_cases else 0.0,
    )
    return EvalReport(
        summary=summary,
        stage_metrics=stage_metrics,
        by_tag=by_tag,
        case_results=case_results,
    )


def _evaluate_case(
    case: EvalCase,
    result: Optional[EvalRunResult],
    stages: Sequence[str],
    k_values: Sequence[int],
    ingestion_index: Optional["_IngestionIndex"] = None,
) -> CaseResult:
    items = result.retrieved_items if result else []
    stage_metrics = {
        stage: _evaluate_stage(case.expected_evidence, _items_for_stage(items, stage), k_values)
        for stage in stages
    }
    return CaseResult(
        case_id=case.case_id,
        tags=case.tags,
        stage_metrics=stage_metrics,
        permission_leaks=_find_permission_leaks(case, items),
        ingestion_gaps=_find_ingestion_gaps(case, ingestion_index),
    )


def _evaluate_stage(
    expected_evidence: Sequence[Evidence],
    retrieved_items: Sequence[RetrievedItem],
    k_values: Sequence[int],
) -> StageMetrics:
    relevant_evidence = _target_evidence(expected_evidence)
    if not relevant_evidence:
        return _evaluate_no_answer_case(retrieved_items, k_values)

    first_match_rank = _first_match_rank(relevant_evidence, retrieved_items)
    hit_at_k: Dict[int, float] = {}
    recall_at_k: Dict[int, float] = {}
    precision_at_k: Dict[int, float] = {}
    ndcg_at_k: Dict[int, float] = {}

    for k in k_values:
        top_items = list(retrieved_items[:k])
        matched_keys = _matched_evidence_keys(relevant_evidence, top_items)
        hit_at_k[k] = 1.0 if matched_keys else 0.0
        recall_at_k[k] = len(matched_keys) / len(relevant_evidence)
        precision_at_k[k] = len(matched_keys) / k
        ndcg_at_k[k] = _ndcg(relevant_evidence, top_items, k)

    return StageMetrics(
        case_count=1,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        ndcg_at_k=ndcg_at_k,
        mrr=1.0 / first_match_rank if first_match_rank else 0.0,
    )


def _evaluate_no_answer_case(
    retrieved_items: Sequence[RetrievedItem],
    k_values: Sequence[int],
) -> StageMetrics:
    hit_at_k = {k: 1.0 if not retrieved_items[:k] else 0.0 for k in k_values}
    recall_at_k = dict(hit_at_k)
    precision_at_k = dict(hit_at_k)
    ndcg_at_k = dict(hit_at_k)
    return StageMetrics(
        case_count=1,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        ndcg_at_k=ndcg_at_k,
        mrr=1.0 if not retrieved_items else 0.0,
    )


def _target_evidence(expected_evidence: Sequence[Evidence]) -> List[Evidence]:
    must = [item for item in expected_evidence if item.importance == "must"]
    return must or list(expected_evidence)


def _items_for_stage(items: Sequence[RetrievedItem], stage: str) -> List[RetrievedItem]:
    return sorted([item for item in items if item.stage == stage], key=lambda item: item.sort_key())


def _first_match_rank(
    expected_evidence: Sequence[Evidence],
    retrieved_items: Sequence[RetrievedItem],
) -> Optional[int]:
    for index, item in enumerate(retrieved_items, start=1):
        if any(_evidence_matches_item(evidence, item) for evidence in expected_evidence):
            return index
    return None


def _matched_evidence_keys(
    expected_evidence: Sequence[Evidence],
    retrieved_items: Sequence[RetrievedItem],
) -> set[str]:
    matched = set()
    for evidence in expected_evidence:
        if any(_evidence_matches_item(evidence, item) for item in retrieved_items):
            matched.add(evidence.key())
    return matched


def _evidence_matches_item(evidence: Evidence, item: RetrievedItem) -> bool:
    if evidence.fragment_id and item.fragment_id and evidence.fragment_id == item.fragment_id:
        return True
    if evidence.quote and _quote_matches_content(evidence.quote, item.content):
        return True
    return False


def _quote_matches_content(quote: str, content: str) -> bool:
    if quote in content:
        return True
    normalized_quote = _normalize_whitespace(quote)
    normalized_content = _normalize_whitespace(content)
    return bool(normalized_quote) and normalized_quote in normalized_content


def _ndcg(expected_evidence: Sequence[Evidence], retrieved_items: Sequence[RetrievedItem], k: int) -> float:
    ideal_relevant_count = min(len(expected_evidence), k)
    if ideal_relevant_count == 0:
        return 0.0

    matched_keys: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(retrieved_items[:k], start=1):
        matched_key = None
        for evidence in expected_evidence:
            if evidence.key() in matched_keys:
                continue
            if _evidence_matches_item(evidence, item):
                matched_key = evidence.key()
                break
        if matched_key:
            matched_keys.add(matched_key)
            dcg += 1.0 / math.log2(rank + 1)

    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _find_permission_leaks(case: EvalCase, items: Sequence[RetrievedItem]) -> List[RetrievedItem]:
    allowed = set(case.kb_ids)
    forbidden = set(case.forbidden_kb_ids)
    leaks: List[RetrievedItem] = []
    seen: set[tuple[str, str, str]] = set()

    for item in items:
        if not item.kb_id:
            continue
        is_forbidden = item.kb_id in forbidden
        is_outside_scope = case.strict_kb_scope and bool(allowed) and item.kb_id not in allowed
        if not is_forbidden and not is_outside_scope:
            continue
        key = (item.stage, item.fragment_id or "", item.kb_id)
        if key in seen:
            continue
        leaks.append(item)
        seen.add(key)
    return leaks


@dataclass(frozen=True)
class _IngestionIndex:
    by_fragment_id: Dict[str, List[IngestedFragment]]
    fragments: List[IngestedFragment]


def _build_ingestion_index(
    fragments: Optional[Sequence[IngestedFragment]],
) -> Optional[_IngestionIndex]:
    if fragments is None:
        return None

    by_fragment_id: Dict[str, List[IngestedFragment]] = {}
    normalized_fragments = list(fragments)
    for fragment in normalized_fragments:
        if not fragment.fragment_id:
            continue
        by_fragment_id.setdefault(fragment.fragment_id, []).append(fragment)

    return _IngestionIndex(
        by_fragment_id=by_fragment_id,
        fragments=normalized_fragments,
    )


def _find_ingestion_gaps(
    case: EvalCase,
    ingestion_index: Optional[_IngestionIndex],
) -> List[Dict[str, Any]]:
    if ingestion_index is None:
        return []

    gaps: List[Dict[str, Any]] = []
    for evidence in _target_evidence(case.expected_evidence):
        matching_fragments = _matching_ingested_fragments(case, evidence, ingestion_index)
        if evidence.fragment_id and not matching_fragments:
            gaps.append(
                {
                    "reason": "missing_fragment",
                    "fragment_id": evidence.fragment_id,
                    "quote": evidence.quote,
                }
            )
            continue

        if evidence.quote and not any(
            _quote_matches_content(evidence.quote, fragment.content)
            for fragment in matching_fragments
        ):
            gaps.append(
                {
                    "reason": "missing_quote",
                    "fragment_id": evidence.fragment_id,
                    "quote": evidence.quote,
                }
            )

    return gaps


def _matching_ingested_fragments(
    case: EvalCase,
    evidence: Evidence,
    ingestion_index: _IngestionIndex,
) -> List[IngestedFragment]:
    candidates = (
        ingestion_index.by_fragment_id.get(evidence.fragment_id, [])
        if evidence.fragment_id
        else ingestion_index.fragments
    )
    scoped_candidates = [
        fragment
        for fragment in candidates
        if _fragment_in_case_scope(case, fragment)
    ]
    if evidence.fragment_id:
        return scoped_candidates
    if not evidence.quote:
        return scoped_candidates
    return [
        fragment
        for fragment in scoped_candidates
        if _quote_matches_content(evidence.quote, fragment.content)
    ]


def _fragment_in_case_scope(case: EvalCase, fragment: IngestedFragment) -> bool:
    if not case.kb_ids:
        return True
    return bool(fragment.kb_id) and fragment.kb_id in set(case.kb_ids)


def _aggregate_stage_metrics(metrics: Sequence[StageMetrics], k_values: Sequence[int]) -> StageMetrics:
    case_count = len(metrics)
    if case_count == 0:
        return StageMetrics(case_count=0)

    return StageMetrics(
        case_count=case_count,
        hit_at_k={k: _average(metric.hit_at_k.get(k, 0.0) for metric in metrics) for k in k_values},
        recall_at_k={k: _average(metric.recall_at_k.get(k, 0.0) for metric in metrics) for k in k_values},
        precision_at_k={k: _average(metric.precision_at_k.get(k, 0.0) for metric in metrics) for k in k_values},
        ndcg_at_k={k: _average(metric.ndcg_at_k.get(k, 0.0) for metric in metrics) for k in k_values},
        mrr=_average(metric.mrr for metric in metrics),
    )


def _collect_stages(results: Mapping[str, EvalRunResult]) -> List[str]:
    stages = {
        item.stage
        for result in results.values()
        for item in result.retrieved_items
        if item.stage
    }
    return sorted(stages or {DEFAULT_STAGE})


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 格式错误：{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL 每一行都必须是对象：{path}:{line_number}")
            records.append(value)
    return records


def _average(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0


def _string_keyed_float_dict(values: Mapping[int, float]) -> Dict[str, float]:
    return {str(key): round(float(value), 6) for key, value in sorted(values.items())}


def _clean_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _normalize_whitespace(value: str) -> str:
    return " ".join(str(value).split())


def _clean_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
