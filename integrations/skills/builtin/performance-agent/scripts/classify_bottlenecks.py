#!/usr/bin/env python3
"""Classify performance bottlenecks from multi-dimensional analysis artifacts.

Produces ranked bottleneck candidates with confidence scores and evidence.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, write_json


def candidate(
    name: str,
    confidence: float,
    evidence: list[str],
    validation_checks: list[str],
    optimization_hints: list[str],
) -> dict:
    return {
        "name": name,
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "validation_checks": validation_checks,
        "optimization_hints": optimization_hints,
    }


def add_candidate(candidates_by_name: dict[str, dict], item: dict) -> None:
    current = candidates_by_name.get(item["name"])
    if not current:
        candidates_by_name[item["name"]] = {
            "name": item["name"],
            "confidence": item["confidence"],
            "evidence": list(item["evidence"]),
            "validation_checks": list(item["validation_checks"]),
            "optimization_hints": list(item["optimization_hints"]),
        }
        return
    merged = {
        "name": item["name"],
        "confidence": round(max(current["confidence"], item["confidence"]), 3),
        "evidence": current["evidence"] + [v for v in item["evidence"] if v not in current["evidence"]],
        "validation_checks": current["validation_checks"] + [v for v in item["validation_checks"] if v not in current["validation_checks"]],
        "optimization_hints": current["optimization_hints"] + [v for v in item["optimization_hints"] if v not in current["optimization_hints"]],
    }
    candidates_by_name[item["name"]] = merged


def classify(
    profile: dict,
    step: Optional[dict],
    communication: Optional[dict],
    memory: Optional[dict],
    input_summary: Optional[dict],
    trace_gaps: Optional[dict],
    hotspot: Optional[dict],
    mfu: Optional[dict] = None,
    cluster: Optional[dict] = None,
    jitter: Optional[dict] = None,
    fusion: Optional[dict] = None,
    degradation: Optional[dict] = None,
    affinity: Optional[dict] = None,
    collective_types: Optional[dict] = None,
    rank_variance: Optional[dict] = None,
) -> list[dict]:
    candidates_by_name: dict[str, dict] = {}

    if communication and communication.get("dominant_collective"):
        pressure = communication.get("communication_pressure")
        base = 0.8 if pressure == "high" else 0.6
        evidence = [
            f"dominant collective: {communication['dominant_collective']['name']}",
            f"communication pressure: {pressure}",
        ]
        if communication.get("matrix_imbalance_ratio") is not None:
            evidence.append(f"matrix imbalance ratio: {communication['matrix_imbalance_ratio']}")
        add_candidate(
            candidates_by_name,
            candidate(
                "communication",
                base,
                evidence,
                ["compare collective time share", "compare collective count", "compare exposed step tail"],
                ["check overlap", "check bucket or fusion settings", "remove unnecessary synchronization"],
            ),
        )

    if input_summary and input_summary.get("bottleneck_detected"):
        evidence = []
        if input_summary.get("queue_empty_percent") is not None:
            evidence.append(f"queue empty percent: {input_summary['queue_empty_percent']}")
        if input_summary.get("warning"):
            evidence.append(f"pipeline warning: {input_summary['warning']}")
        add_candidate(
            candidates_by_name,
            candidate(
                "input_pipeline",
                0.7,
                evidence,
                ["compare pre-compute idle time", "compare queue-empty rate", "compare end-to-end throughput"],
                ["increase pipeline parallelism", "reduce decode or transform cost", "check prefetch or caching"],
            ),
        )

    if memory and memory.get("peak_memory_mb") is not None:
        evidence = [f"peak memory: {memory['peak_memory_mb']} MB"]
        if memory.get("top_operators"):
            top = memory["top_operators"][0]
            evidence.append(f"top memory operator: {top['name']} ({top['share_percent']}%)")
        add_candidate(
            candidates_by_name,
            candidate(
                "memory",
                0.75 if memory.get("memory_pressure") == "high" else 0.55,
                evidence,
                ["compare peak memory", "compare top memory-consuming stage", "compare batch-size headroom"],
                ["review recomputation", "reduce temporary tensors", "review precision and layout"],
            ),
        )

    if trace_gaps and trace_gaps.get("dominant_category"):
        dominant = trace_gaps["dominant_category"]["name"]
        share = float(trace_gaps["dominant_category"].get("share_percent") or 0.0)
        evidence = [f"dominant trace category: {dominant}", f"share: {round(share, 2)}%"]
        if dominant in {"host_overhead", "idle_gap"}:
            add_candidate(
                candidates_by_name,
                candidate(
                    "host_framework_overhead",
                    0.72 if share >= 20 else 0.6,
                    evidence,
                    ["compare trace idle gap", "compare host launch duration", "compare kernel launch density"],
                    ["reduce host-side per-step work", "remove unnecessary syncs", "increase graph-heavy execution"],
                ),
            )
        elif dominant == "graph_compile":
            add_candidate(
                candidates_by_name,
                candidate(
                    "graph_compile",
                    0.72 if share >= 20 else 0.6,
                    evidence,
                    ["compare compile duration", "compare compile count", "compare warmup versus steady-state latency"],
                    ["stabilize shapes", "reduce recompilation triggers", "separate warmup compile from steady-state measurement"],
                ),
            )
        elif dominant == "communication":
            add_candidate(
                candidates_by_name,
                candidate(
                    "communication",
                    0.7 if share >= 20 else 0.58,
                    evidence,
                    ["compare communication slices in the trace", "compare overlap between communication and compute", "compare exposed step tail"],
                    ["check overlap", "check bucket or fusion settings", "remove unnecessary synchronization"],
                ),
            )
        elif dominant == "input_pipeline":
            add_candidate(
                candidates_by_name,
                candidate(
                    "input_pipeline",
                    0.68 if share >= 20 else 0.55,
                    evidence,
                    ["compare queue-empty rate", "compare dataset stage time", "compare pre-compute idle time"],
                    ["increase pipeline parallelism", "reduce decode or transform cost", "check prefetch or caching"],
                ),
            )

    if step and step.get("dominant_stage"):
        dominant = step["dominant_stage"]["name"]
        share = step["dominant_stage"]["share_percent"]
        if dominant in {"host_overhead", "idle_gap"}:
            add_candidate(
                candidates_by_name,
                candidate(
                    "host_framework_overhead",
                    0.68,
                    [f"dominant stage: {dominant}", f"share: {share}%"],
                    ["compare host idle gap", "compare kernel launch density", "compare utilization trend"],
                    ["reduce host-side per-step work", "remove unnecessary syncs", "increase graph-heavy execution"],
                ),
            )
        elif dominant == "graph_compile":
            add_candidate(
                candidates_by_name,
                candidate(
                    "graph_compile",
                    0.66,
                    [f"dominant stage: {dominant}", f"share: {share}%"],
                    ["compare compile time", "compare compile count", "compare steady-state latency after warmup"],
                    ["stabilize shapes", "reduce recompilation triggers", "separate first-run compile from steady state"],
                ),
            )
        elif dominant == "compute":
            add_candidate(
                candidates_by_name,
                candidate(
                    "compute",
                    0.52,
                    [f"dominant stage: {dominant}", f"share: {share}%"],
                    ["compare operator time share", "compare end-to-end step time"],
                    ["focus on top compute operators", "review fusion opportunities", "review backend kernel path"],
                ),
            )

    if hotspot and hotspot.get("top_operators"):
        lead = hotspot["top_operators"][0]
        confidence = 0.7 if float(lead.get("share_percent") or 0.0) >= 35 else 0.5
        hints = ["focus on the top operator first", "review fusion or kernel path", "avoid spreading effort across the long tail"]
        if lead.get("category") == "communication":
            hints = ["check overlap", "check collective fusion", "validate communication-heavy update or step tail"]
        add_candidate(
            candidates_by_name,
            candidate(
                "operator_hotspot",
                confidence,
                [f"lead operator: {lead['operator']}", f"time share: {lead['share_percent']}%"],
                ["compare dominant operator share", "compare end-to-end step time or latency"],
                hints,
            ),
        )

    # New: Low MFU bottleneck
    if mfu and mfu.get("estimated_mfu") is not None:
        mfu_val = mfu["estimated_mfu"]
        if mfu_val < 0.30:
            add_candidate(
                candidates_by_name,
                candidate(
                    "low_mfu",
                    0.75 if mfu_val < 0.20 else 0.65,
                    [f"estimated MFU: {mfu_val*100:.1f}%", f"MFU level: {mfu.get('mfu_level', 'unknown')}"],
                    ["compare MFU after enabling graph compilation", "compare operator fusion coverage"],
                    ["enable graph compilation (GRAPH_MODE / torch.compile)", "check for excessive small operators", "enable operator fusion"],
                ),
            )

    # New: Rank imbalance
    if cluster and cluster.get("slow_ranks"):
        analysis = cluster.get("analysis", {})
        bt_type = analysis.get("bottleneck_type", "general")
        slow_ranks = cluster["slow_ranks"]
        add_candidate(
            candidates_by_name,
            candidate(
                "rank_imbalance",
                0.78,
                [f"slow ranks: {slow_ranks}", f"bottleneck type: {bt_type}", analysis.get("diagnosis", "")],
                ["compare per-rank step times", "compare operator stats between slow and fast ranks"],
                [
                    f"investigate Rank {slow_ranks[0]} ({bt_type})",
                    "check CPU affinity and NUMA binding",
                    "compare API dispatch stats" if bt_type == "host_dispatch" else "compare operator stats",
                ],
            ),
        )

    # New: Jitter
    if jitter:
        step_jitter = jitter.get("step_time_jitter", {})
        cv = step_jitter.get("cv")
        if cv and cv > 0.15:
            add_candidate(
                candidates_by_name,
                candidate(
                    "jitter",
                    0.60 if cv > 0.20 else 0.50,
                    [f"step time CV: {cv*100:.1f}%", f"status: {step_jitter.get('status')}"],
                    ["compare step time CV after fixes", "compare outlier count"],
                    ["pad sequences to fixed lengths", "enable CPU affinity", "check for GC pauses"],
                ),
            )

    # New: Fusion opportunity
    if fusion and fusion.get("fusion_analysis_available"):
        opportunities = fusion.get("opportunities", [])
        if opportunities:
            top_opp = opportunities[0]
            combined_share = top_opp.get("combined_share_percent", 0)
            add_candidate(
                candidates_by_name,
                candidate(
                    "fusion_opportunity",
                    0.70 if combined_share >= 20 else 0.55,
                    [f"fusion_type: {top_opp.get('type', 'unknown')}", f"combined_share: {combined_share}%"],
                    [f"compare {top_opp.get('type', 'unknown')} operator time after fusion"],
                    [f"apply {top_opp.get('replacement_api', 'fused variant')}", top_opp.get("description", "enable operator fusion")],
                ),
            )

    # New: Cluster degradation
    if degradation and degradation.get("degradation_classification_available"):
        primary_type = degradation.get("primary_type")
        sub = degradation.get("sub_classification", {})
        if primary_type:
            add_candidate(
                candidates_by_name,
                candidate(
                    "cluster_degradation",
                    sub.get("confidence", 0.60),
                    [f"degradation_type: {primary_type}", f"likely_cause: {sub.get('likely_cause', 'unknown')}"],
                    degradation.get("evidence", [])[:3],
                    degradation.get("recommended_actions", [])[:3],
                ),
            )

    # New: NPU affinity gap
    if affinity and affinity.get("npu_affinity_analysis_available"):
        score = affinity.get("overall_affinity_score", 1.0)
        if score < 0.8:
            total_findings = affinity.get("total_findings", 0)
            add_candidate(
                candidates_by_name,
                candidate(
                    "npu_affinity_gap",
                    0.65 if score < 0.6 else 0.50,
                    [f"affinity_score: {score}", f"total_findings: {total_findings}"],
                    ["compare affinity score after applying four-step fixes"],
                    [affinity.get("priority_fix", "Apply NPU affinity four-step optimization")],
                ),
            )

    # New: SyncBN synchronization bottleneck
    if collective_types and collective_types.get("collective_type_analysis_available"):
        syncbn_share = collective_types.get("syncbn_share_percent", 0)
        syncbn_dominant = collective_types.get("syncbn_dominant", False)
        if syncbn_share > 10:
            evidence = [f"SyncBN share: {syncbn_share:.1f}%"]
            if syncbn_dominant:
                evidence.append("SyncBN is dominant collective type")
            add_candidate(
                candidates_by_name,
                candidate(
                    "syncbn_synchronization",
                    0.78 if syncbn_dominant else 0.60,
                    evidence,
                    ["compare SyncBN time share after switching to GroupNorm"],
                    [
                        "Consider replacing SyncBN with GroupNorm or FrozenBN",
                        "Reduce SyncBN synchronization frequency if possible",
                        "Check if batch normalization can be computed locally",
                    ],
                ),
            )

    # New: Rank compute jitter
    if rank_variance and rank_variance.get("rank_variance_analysis_available"):
        jittery_ranks = rank_variance.get("jittery_ranks", [])
        worst_cv = rank_variance.get("worst_rank_cv", 0)
        drag = rank_variance.get("drag_effect_ms", 0)
        if jittery_ranks and worst_cv > 0.10:
            rank_str = ", ".join(str(r) for r in jittery_ranks[:3])
            add_candidate(
                candidates_by_name,
                candidate(
                    "rank_compute_jitter",
                    0.75 if worst_cv > 0.20 else 0.60,
                    [f"jittery ranks: [{rank_str}]", f"worst CV: {worst_cv:.3f}", f"drag effect: {drag:.1f}ms"],
                    ["compare per-rank CV after stabilizing compute"],
                    [
                        "Investigate compute instability on jittery rank(s)",
                        "Check for dynamic shapes, GC pauses, or CPU scheduling",
                        "Enable CPU affinity (numactl/taskset)",
                    ],
                ),
            )

    candidates = list(candidates_by_name.values())

    if not candidates:
        candidates.append(
            candidate(
                "inconclusive",
                0.2,
                ["structured summaries are insufficient for a bottleneck claim"],
                ["collect step, operator, and trace summaries"],
                ["collect stronger profiler evidence before changing the workload"],
            )
        )

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify ranked bottleneck candidates from structured summaries")
    parser.add_argument("--profile-json", required=True, help="performance profile JSON path")
    parser.add_argument("--step-json", help="step summary JSON path")
    parser.add_argument("--communication-json", help="communication summary JSON path")
    parser.add_argument("--memory-json", help="memory summary JSON path")
    parser.add_argument("--input-json", help="input summary JSON path")
    parser.add_argument("--trace-gaps-json", help="trace-gap summary JSON path")
    parser.add_argument("--hotspot-json", help="hotspot summary JSON path")
    parser.add_argument("--mfu-json", help="MFU calculation JSON path")
    parser.add_argument("--cluster-json", help="cluster analysis JSON path")
    parser.add_argument("--jitter-json", help="jitter analysis JSON path")
    parser.add_argument("--fusion-json", help="operator fusion analysis JSON path")
    parser.add_argument("--degradation-json", help="cluster degradation classification JSON path")
    parser.add_argument("--affinity-json", help="NPU affinity analysis JSON path")
    parser.add_argument("--collective-types-json", help="Collective type analysis JSON path")
    parser.add_argument("--rank-variance-json", help="Rank variance analysis JSON path")
    parser.add_argument("--output-json", required=True, help="path to write the bottleneck classification JSON")
    args = parser.parse_args()

    try:
        profile = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading profile JSON: {e}", file=__import__("sys").stderr)
        return 1
    step = load_optional_json(args.step_json)
    communication = load_optional_json(args.communication_json)
    memory = load_optional_json(args.memory_json)
    input_summary = load_optional_json(args.input_json)
    trace_gaps = load_optional_json(args.trace_gaps_json)
    hotspot = load_optional_json(args.hotspot_json)
    mfu = load_optional_json(args.mfu_json)
    cluster = load_optional_json(args.cluster_json)
    jitter = load_optional_json(args.jitter_json)
    fusion = load_optional_json(args.fusion_json)
    degradation = load_optional_json(args.degradation_json)
    affinity = load_optional_json(args.affinity_json)
    collective_types = load_optional_json(args.collective_types_json)
    rank_variance = load_optional_json(args.rank_variance_json)

    ranked = classify(profile, step, communication, memory, input_summary, trace_gaps, hotspot, mfu, cluster, jitter, fusion, degradation, affinity, collective_types, rank_variance)
    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "primary_candidate": ranked[0],
        "ranked_candidates": ranked,
        "next_action": (
            "Apply one targeted optimization and collect a before/after comparison for the primary candidate."
            if ranked[0]["name"] != "inconclusive"
            else "Collect stronger profiler evidence before choosing the first optimization."
        ),
    }
    write_json(Path(args.output_json), report)
    print(json.dumps({"primary_candidate": report["primary_candidate"]["name"], "confidence": report["primary_candidate"]["confidence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
