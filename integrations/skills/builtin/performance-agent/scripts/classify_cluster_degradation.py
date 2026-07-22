#!/usr/bin/env python3
"""Classify cluster performance degradation patterns.

Identifies degradation types: scale-up, hardware change, long-term training,
performance fluctuation, and slow node vs network problems.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, write_json


# Degradation classification rules.
DEGRADATION_TYPES = {
    "scale_up": {
        "name": "Scale-Up Degradation",
        "description": "Performance degradation when scaling from small to large cluster",
        "likely_cause": "model_sharding_strategy",
        "actions": [
            "Review TP/PP/DP configuration for the larger cluster",
            "Check if communication volume scales linearly with cluster size",
            "Consider increasing micro batch count for better overlap",
            "Evaluate whether ZeRO stage is appropriate for this scale",
        ],
    },
    "hardware_change": {
        "name": "Hardware Change Degradation",
        "description": "Performance regression after hardware replacement or reconfiguration",
        "likely_cause": "hardware_mismatch",
        "actions": [
            "Compare before/after profiler data at component level",
            "Check firmware and driver versions on new hardware",
            "Verify network topology and link configuration",
            "Run hardware diagnostic benchmarks",
        ],
    },
    "long_term_training": {
        "name": "Long-Term Training Degradation",
        "description": "Gradual or sudden performance drop during extended training",
        "likely_cause": "memory_leak_or_thermal",
        "actions": [
            "Plot step time over training duration to detect trend",
            "Check for memory growth across steps (potential leak)",
            "Monitor NPU thermal throttling indicators",
            "Check for OS-level resource exhaustion (file descriptors, memory)",
        ],
    },
    "performance_fluctuation": {
        "name": "Performance Fluctuation",
        "description": "Intermittent performance variation with no consistent degradation",
        "likely_cause": "resource_contention_or_input_variance",
        "actions": [
            "Correlate fluctuation steps with specific phases",
            "Check for OS scheduling interference (CPU affinity, NUMA)",
            "Verify input data characteristics (variable-length sequences)",
            "Monitor for GC pauses in Python runtime",
        ],
    },
    "slow_node": {
        "name": "Slow Node (Asymmetric)",
        "description": "One or more specific nodes are slower, causing others to wait",
        "likely_cause": "node_hardware_or_config_issue",
        "actions": [
            "Compare API dispatch stats between slow and fast nodes",
            "Check CPU affinity and NUMA binding on the slow node",
            "Run hardware diagnostics on the affected node",
            "Compare operator-level timing between slow and fast nodes",
        ],
    },
    "network_problem": {
        "name": "Network Problem (All Cards Affected)",
        "description": "All cards show degraded performance due to network issues",
        "likely_cause": "network_congestion_or_misconfiguration",
        "actions": [
            "Check PFC queue anomalies on network switches",
            "Verify RDMA/RoCE configuration and link health",
            "Check for network congestion during the affected period",
            "Validate HCCL buffer sizes for the cluster configuration",
        ],
    },
}


def _has_linearity_issue(linearity_json: Optional[dict]) -> Optional[float]:
    """Check if linearity is below threshold (0.8)."""
    if not linearity_json:
        return None
    linearity = linearity_json.get("linearity")
    if linearity is not None:
        try:
            return float(linearity)
        except (ValueError, TypeError):
            pass
    return None


def _has_high_comm_pressure(comm_json: Optional[dict]) -> bool:
    """Check if communication pressure is high."""
    if not comm_json:
        return False
    return comm_json.get("communication_pressure") in ("high", "moderate")


def _has_jitter_issue(jitter_json: Optional[dict]) -> Optional[float]:
    """Check if jitter is significant (CV > 0.15)."""
    if not jitter_json:
        return None
    step_jitter = jitter_json.get("step_time_jitter", {})
    cv = step_jitter.get("cv")
    if cv is not None:
        try:
            return float(cv)
        except (ValueError, TypeError):
            pass
    return None


def _has_slow_ranks(cluster_json: Optional[dict]) -> list:
    """Get list of slow ranks if any."""
    if not cluster_json:
        return []
    return cluster_json.get("slow_ranks", [])


def _has_asymmetric_pattern(cluster_json: Optional[dict]) -> bool:
    """Check if slow ranks show asymmetric pattern (only some affected)."""
    slow_ranks = _has_slow_ranks(cluster_json)
    if not slow_ranks:
        return False
    # If only some ranks are slow, it's asymmetric
    analysis = cluster_json.get("analysis", {})
    bottleneck_type = analysis.get("bottleneck_type", "")
    return bottleneck_type in ("host_dispatch", "compute")


def classify_degradation(
    cluster_json: Optional[dict],
    jitter_json: Optional[dict],
    step_json: Optional[dict],
    comm_json: Optional[dict],
    linearity_json: Optional[dict],
) -> dict:
    """Classify cluster performance degradation."""
    if not cluster_json and not jitter_json and not step_json:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "degradation_classification_available": False,
        }

    linearity = _has_linearity_issue(linearity_json)
    high_comm = _has_high_comm_pressure(comm_json)
    jitter_cv = _has_jitter_issue(jitter_json)
    slow_ranks = _has_slow_ranks(cluster_json)
    asymmetric = _has_asymmetric_pattern(cluster_json)

    evidence = []
    classifications = []

    # Evidence collection
    if linearity is not None and linearity < 0.8:
        evidence.append(f"linearity={linearity:.2f} (<0.8 threshold)")
    if high_comm:
        evidence.append("communication_pressure=high")
    if jitter_cv is not None and jitter_cv > 0.15:
        evidence.append(f"step_time_cv={jitter_cv:.2f} (>0.15 threshold)")
    if slow_ranks:
        evidence.append(f"slow_ranks={slow_ranks}")

    # Classification logic

    # Scale-up degradation: low linearity + high comm pressure + no slow ranks
    if (linearity is not None and linearity < 0.8
            and high_comm and not slow_ranks):
        classifications.append({
            "type": "scale_up",
            "confidence": 0.78,
            "reason": "Low linearity with high communication pressure and no slow ranks indicates scaling bottleneck",
        })

    # Slow node: asymmetric slow ranks
    if asymmetric and slow_ranks:
        classifications.append({
            "type": "slow_node",
            "confidence": 0.80,
            "reason": f"Asymmetric performance with slow ranks {slow_ranks} indicates node-specific issue",
        })

    # Network problem: all cards affected + communication pressure
    if slow_ranks and not asymmetric and high_comm:
        classifications.append({
            "type": "network_problem",
            "confidence": 0.75,
            "reason": "All cards affected with high communication pressure indicates network issue",
        })

    # Performance fluctuation: high jitter without consistent degradation
    if jitter_cv is not None and jitter_cv > 0.15:
        classifications.append({
            "type": "performance_fluctuation",
            "confidence": 0.65 if jitter_cv < 0.25 else 0.80,
            "reason": f"Step time CV={jitter_cv:.2f} indicates significant performance variance",
        })

    # Long-term training: requires step time trend data
    if step_json:
        raw_times = step_json.get("step_times", [])
        # Safely convert to float, filtering invalid values
        try:
            step_times = [float(t) for t in raw_times if t is not None]
        except (ValueError, TypeError):
            step_times = []
        if len(step_times) >= 10:
            mid = len(step_times) // 2
            first_half_avg = sum(step_times[:mid]) / mid
            second_half_avg = sum(step_times[mid:]) / (len(step_times) - mid)
            if first_half_avg > 0 and second_half_avg > first_half_avg * 1.1:
                pct_increase = ((second_half_avg / first_half_avg) - 1) * 100
                classifications.append({
                    "type": "long_term_training",
                    "confidence": 0.70,
                    "reason": f"Step time trend: {first_half_avg:.1f}ms -> {second_half_avg:.1f}ms (+{pct_increase:.1f}%)",
                })

    # Sort by confidence
    classifications.sort(key=lambda c: c["confidence"], reverse=True)

    # If no classifications matched any pattern, report unavailable
    if not classifications:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "degradation_classification_available": False,
        }

    # Determine primary type
    primary = classifications[0] if classifications else None

    # Get detailed info for primary type
    sub_classification = None
    recommended_actions = []
    if primary:
        type_key = primary["type"]
        type_info = DEGRADATION_TYPES.get(type_key, {})
        sub_classification = {
            "type": type_key,
            "name": type_info.get("name", type_key),
            "likely_cause": type_info.get("likely_cause", "unknown"),
            "confidence": primary["confidence"],
            "reason": primary["reason"],
        }
        recommended_actions = type_info.get("actions", [])

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "degradation_classification_available": True,
        "primary_type": primary["type"] if primary else None,
        "all_types": [c["type"] for c in classifications],
        "evidence": evidence,
        "sub_classification": sub_classification,
        "recommended_actions": recommended_actions,
        "classifications": classifications,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify cluster performance degradation patterns"
    )
    parser.add_argument("--cluster-json", help="Cluster analysis JSON path")
    parser.add_argument("--jitter-json", help="Jitter analysis JSON path")
    parser.add_argument("--step-json", help="Step breakdown JSON path")
    parser.add_argument("--communication-json", help="Communication summary JSON path")
    parser.add_argument("--linearity-json", help="Linearity calculation JSON path")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    cluster = load_optional_json(args.cluster_json)
    jitter = load_optional_json(args.jitter_json)
    step = load_optional_json(args.step_json)
    comm = load_optional_json(args.communication_json)
    linearity = load_optional_json(args.linearity_json)

    result = classify_degradation(cluster, jitter, step, comm, linearity)

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "classification_available": result.get("degradation_classification_available", False),
        "primary_type": result.get("primary_type"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
