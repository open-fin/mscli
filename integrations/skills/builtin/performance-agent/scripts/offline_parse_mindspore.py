#!/usr/bin/env python3
"""Offline parser for MindSpore Profiler data (*_ascend_ms).

Calls mindspore.profiler.profiler.analyse() to generate ASCEND_PROFILER_OUTPUT
from raw profiler data that has not yet been parsed.

Prerequisites:
  - mindspore installed
  - msprof in PATH (source /usr/local/Ascend/ascend-toolkit/set_env.sh)
"""
import argparse
import sys
from pathlib import Path

from perf_common import check_msprof_available


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline parser for MindSpore Profiler data (*_ascend_ms)."
    )
    parser.add_argument(
        "profiler_path",
        type=str,
        help="Path to the profiler result directory (e.g., ./result_data_ascend_ms)",
    )
    args = parser.parse_args()

    profiler_path = Path(args.profiler_path).resolve()
    if not profiler_path.exists():
        print(f"Error: Profiler path '{profiler_path}' does not exist.", file=sys.stderr)
        return 1

    # Check if already parsed
    if (profiler_path / "ASCEND_PROFILER_OUTPUT").exists():
        print(f"ASCEND_PROFILER_OUTPUT already exists at {profiler_path}. Skipping parse.")
        return 0

    check_msprof_available()

    try:
        from mindspore.profiler.profiler import analyse  # type: ignore[import-untyped]
    except ImportError as e:
        print(f"Error: MindSpore not installed or analyse unavailable: {e}", file=sys.stderr)
        return 1

    print(f"Starting offline analysis for: {profiler_path}")
    try:
        analyse(profiler_path=str(profiler_path))
        print("Analysis completed successfully.")
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
