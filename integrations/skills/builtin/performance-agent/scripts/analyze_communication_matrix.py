#!/usr/bin/env python3
"""Analyze communication matrix for link-level bandwidth and topology insights.

Loads communication_matrix.json and classifies link types (HCCS intra-node,
RDMA inter-node, PCIe cross-ring), detects slow links via z-score, and
generates HCCL tuning suggestions.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from perf_common import normalize_key, parse_number, read_json, write_json

# Bandwidth expectations in GB/s
_HCCS_EXPECTED_GB_S = 56.0
_PCIE_EXPECTED_GB_S = 28.0
_RDMA_MIN_GB_S = 10.0
_RDMA_MAX_GB_S = 25.0

# Slow link z-score threshold
_SLOW_LINK_Z_THRESHOLD = -2.0


def classify_link_type(
    src_rank: int, dst_rank: int, world_size: int
) -> str:
    """Classify the communication link type based on rank placement.

    Assumes standard 8-NPU topology:
    - Ranks 0-3: Ring 0 (HCCS)
    - Ranks 4-7: Ring 1 (HCCS)
    - Cross-ring: PCIe (~28 GB/s, half bandwidth)

    For >8 NPUs, cross-node links are RDMA/RoCE.
    """
    npus_per_node = 8
    src_node = src_rank // npus_per_node
    dst_node = dst_rank // npus_per_node

    if src_node == dst_node:
        # Intra-node
        src_ring = (src_rank % npus_per_node) // 4
        dst_ring = (dst_rank % npus_per_node) // 4
        if src_ring == dst_ring:
            return "hccs_intra_ring"
        return "pcie_cross_ring"
    return "rdma_inter_node"


def extract_link_bandwidths(matrix_data: object) -> list[dict]:
    """Extract per-link bandwidth data from communication matrix JSON.

    Handles multiple JSON formats: nested dict, flat array, or typed records.
    Returns list of dicts with src, dst, bandwidth_gb_s, link_type, transit_time_ms.
    """
    links: list[dict] = []

    if isinstance(matrix_data, list):
        for item in matrix_data:
            if not isinstance(item, dict):
                continue
            link = _extract_single_link(item)
            if link:
                links.append(link)
    elif isinstance(matrix_data, dict):
        # Try nested structures
        for key, value in matrix_data.items():
            nkey = normalize_key(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        link = _extract_single_link(item)
                        if link:
                            links.append(link)
            elif isinstance(value, dict):
                link = _extract_single_link(value)
                if link:
                    link["relation"] = nkey
                    links.append(link)

        # If still no links, try flat extraction
        if not links:
            link = _extract_single_link(matrix_data)
            if link:
                links.append(link)

    return links


def _extract_single_link(record: dict) -> Optional[dict]:
    """Extract bandwidth data from a single link record."""
    normalized = {normalize_key(k): v for k, v in record.items()}

    # Find src/dst rank identifiers
    src = None
    dst = None
    for key in ("src_rank", "src", "from_rank", "from", "send"):
        if key in normalized:
            src = parse_number(normalized[key])
            if src is not None:
                src = int(src)
                break
    for key in ("dst_rank", "dst", "to_rank", "to", "recv"):
        if key in normalized:
            dst = parse_number(normalized[key])
            if dst is not None:
                dst = int(dst)
                break

    # Find bandwidth
    bandwidth_gb_s = None
    for key in ("bandwidth", "bandwidth_gb_s", "bw", "speed", "throughput"):
        if key in normalized:
            bandwidth_gb_s = parse_number(normalized[key])
            if bandwidth_gb_s is not None:
                break

    # Find transit time
    transit_time_ms = None
    for key in ("transit_time", "transit_time_ms", "time_ms", "duration_ms", "latency_ms"):
        if key in normalized:
            transit_time_ms = parse_number(normalized[key])
            if transit_time_ms is not None:
                break

    # Find data size
    data_size_mb = None
    for key in ("data_size", "data_size_mb", "size_mb", "msg_size"):
        if key in normalized:
            data_size_mb = parse_number(normalized[key])
            if data_size_mb is not None:
                break

    # Compute bandwidth from size/time if not directly available
    if bandwidth_gb_s is None and data_size_mb is not None and transit_time_ms is not None and transit_time_ms > 0:
        # data_size_mb / (transit_time_ms / 1000) = MB/s → GB/s
        bandwidth_gb_s = data_size_mb / (transit_time_ms / 1000.0) / 1024.0

    if bandwidth_gb_s is None:
        return None

    return {
        "src_rank": src,
        "dst_rank": dst,
        "bandwidth_gb_s": round(bandwidth_gb_s, 3),
        "transit_time_ms": transit_time_ms,
        "data_size_mb": data_size_mb,
    }


def detect_slow_links(links: list[dict], world_size: int) -> list[dict]:
    """Detect slow links using z-score analysis within each link type group.

    Groups links by type (HCCS, PCIe, RDMA) and flags those with
    z-score below threshold within their group.
    """
    # Group by link type
    groups: dict[str, list[dict]] = {}
    for link in links:
        if link.get("src_rank") is not None and link.get("dst_rank") is not None:
            link_type = classify_link_type(link["src_rank"], link["dst_rank"], world_size)
            link["link_type"] = link_type
            groups.setdefault(link_type, []).append(link)
        else:
            link["link_type"] = "unknown"
            groups.setdefault("unknown", []).append(link)

    slow_links: list[dict] = []

    for link_type, group_links in groups.items():
        bandwidths = [link["bandwidth_gb_s"] for link in group_links]
        if len(bandwidths) < 2:
            continue

        mean_bw = sum(bandwidths) / len(bandwidths)
        variance = sum((b - mean_bw) ** 2 for b in bandwidths) / len(bandwidths)
        std_bw = math.sqrt(variance)

        if std_bw <= 0:
            continue

        expected = _expected_bandwidth(link_type)
        for link, bw in zip(group_links, bandwidths):
            z_score = (bw - mean_bw) / std_bw
            link["z_score"] = round(z_score, 3)
            link["expected_bandwidth_gb_s"] = expected
            link["deviation_percent"] = round((bw - expected) / expected * 100, 1) if expected else None

            if z_score < _SLOW_LINK_Z_THRESHOLD:
                slow_links.append({
                    "src_rank": link["src_rank"],
                    "dst_rank": link["dst_rank"],
                    "link_type": link_type,
                    "bandwidth_gb_s": bw,
                    "expected_gb_s": expected,
                    "z_score": round(z_score, 3),
                    "deviation_percent": link["deviation_percent"],
                    "severity": "critical" if bw < expected * 0.5 else "warning",
                })

    return slow_links


def _expected_bandwidth(link_type: str) -> Optional[float]:
    """Return expected bandwidth for a link type."""
    mapping = {
        "hccs_intra_ring": _HCCS_EXPECTED_GB_S,
        "pcie_cross_ring": _PCIE_EXPECTED_GB_S,
        "rdma_inter_node": _RDMA_MAX_GB_S,
    }
    return mapping.get(link_type)


def suggest_hccl_tuning(
    slow_links: list[dict],
    link_type_summary: dict[str, dict],
    world_size: int,
) -> list[str]:
    """Generate HCCL tuning suggestions based on analysis."""
    suggestions: list[str] = []

    # Check for RDMA issues
    rdma_slow = [l for l in slow_links if l["link_type"] == "rdma_inter_node"]
    if rdma_slow:
        suggestions.append(
            f"RDMA slow links detected ({len(rdma_slow)} links): "
            "check RoCE network congestion, PFC queue anomalies, and HCCL_RDMA_TC alignment with switch QoS"
        )

    # Check for cross-ring issues
    pcie_slow = [l for l in slow_links if l["link_type"] == "pcie_cross_ring"]
    if pcie_slow:
        suggestions.append(
            f"PCIe cross-ring slow links detected ({len(pcie_slow)} links): "
            "this is expected for cross-ring communication (~half HCCS bandwidth); "
            "consider HCCL_INTRA_ROCE_ENABLE=1 for 16P setups"
        )

    # Check for intra-ring HCCS issues
    hccs_slow = [l for l in slow_links if l["link_type"] == "hccs_intra_ring"]
    if hccs_slow:
        suggestions.append(
            f"HCCS intra-ring slow links detected ({len(hccs_slow)} links): "
            "verify HCCS link health, check for hardware degradation, "
            "and ensure balanced topology placement"
        )

    # General HCCL buffer suggestion
    if world_size > 1:
        suggestions.append(
            "Consider calculating HCCL_BUFFSIZE = ceil(MBS × S × H × dtype_size / 8MB) "
            "for LLM workloads (default 200MB may not be optimal)"
        )

    return suggestions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze communication matrix for link-level bandwidth and topology insights"
    )
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--matrix-json", help="explicit communication_matrix.json path")
    parser.add_argument("--output-json", required=True, help="output JSON path")
    args = parser.parse_args()

    # Resolve matrix path
    matrix_path = None
    if args.matrix_json:
        matrix_path = Path(args.matrix_json).resolve()
    elif args.trace_root:
        candidate = Path(args.trace_root).resolve() / "ASCEND_PROFILER_OUTPUT" / "communication_matrix.json"
        if candidate.exists():
            matrix_path = candidate

    if not matrix_path or not matrix_path.exists():
        print("communication_matrix.json not found. Provide --matrix-json or --trace-root.", file=sys.stderr)
        raise SystemExit(1)

    matrix_data = read_json(matrix_path)
    links = extract_link_bandwidths(matrix_data)

    if not links:
        print("No link bandwidth data could be extracted from the matrix.", file=sys.stderr)
        raise SystemExit(1)

    # Infer world_size from rank IDs
    all_ranks = set()
    for link in links:
        if link.get("src_rank") is not None:
            all_ranks.add(link["src_rank"])
        if link.get("dst_rank") is not None:
            all_ranks.add(link["dst_rank"])
    world_size = len(all_ranks) if all_ranks else 1

    # Detect slow links
    slow_links = detect_slow_links(links, world_size)

    # Summarize by link type
    link_type_summary: dict[str, dict] = {}
    for link in links:
        lt = link.get("link_type", "unknown")
        lt_summary = link_type_summary.setdefault(lt, {"bandwidths": []})
        bw_list = lt_summary.setdefault("bandwidths", [])
        bw_list.append(link["bandwidth_gb_s"])

    for lt, data in link_type_summary.items():
        bws = data["bandwidths"]
        data["mean_gb_s"] = round(sum(bws) / len(bws), 3) if bws else None
        data["min_gb_s"] = round(min(bws), 3) if bws else None
        data["max_gb_s"] = round(max(bws), 3) if bws else None
        data["count"] = len(bws)
        data["expected_gb_s"] = _expected_bandwidth(lt)
        del data["bandwidths"]

    # Generate suggestions
    suggestions = suggest_hccl_tuning(slow_links, link_type_summary, world_size)

    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "source_file": str(matrix_path),
        "world_size": world_size,
        "total_links_analyzed": len(links),
        "link_type_summary": link_type_summary,
        "slow_links": slow_links,
        "slow_link_count": len(slow_links),
        "hccl_suggestions": suggestions,
        "likely_domains": ["communication"] if slow_links else [],
        "next_action": (
            f"Investigate {len(slow_links)} slow link(s) before adjusting compute kernels."
            if slow_links
            else "Communication link bandwidth is within expected range."
        ),
    }

    write_json(Path(args.output_json), report)
    print(json.dumps({
        "total_links": len(links),
        "slow_links": len(slow_links),
        "link_types": list(link_type_summary.keys()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
