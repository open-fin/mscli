"""
Probe script for PTA op support range.

**This is a TEMPLATE example** targeting torch_npu.npu_sparse_flash_attention.
When adapting to other operators, copy this file and modify:
  - `run_case()`: input tensor construction and API call
  - `main()`: test matrix (dtypes, layouts, parameter combinations)
  - Output field names in CaseResult

What it does:
- Prints environment/version info (torch/torch_npu + relevant env vars).
- Runs a matrix of small test cases across:
  - dtype: float16 / bfloat16
  - layouts: BSND/BSND, TND/TND, BSND/PA_BSND
  - sparse_size: try {16, 128, 2048} (CANN 8.5 may require 2048)
  - attention_mode / return_softmax_lse combinations
- Records success/failure and error messages into a JSON file.

Notes:
- Requires NPU runtime + torch_npu installed.
- By default tries to force PTA to run ACLNN path:
  torch.npu.set_compile_mode(jit_compile=False) if available.
- Use --quick for a reduced test matrix (core combinations only).
- Each test case has a configurable timeout (default 60s) to avoid hangs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-case timeout (Unix only; Windows gracefully degrades to no timeout)
# ---------------------------------------------------------------------------
class CaseTimeoutError(Exception):
    pass


@contextmanager
def case_timeout(seconds: int) -> Generator[None, None, None]:
    """Context manager that raises CaseTimeoutError after *seconds*."""
    if sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        # signal.alarm not available on Windows; skip timeout
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise CaseTimeoutError(f"Case timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@dataclass
class CaseResult:
    name: str
    config: Dict[str, Any]
    ok: bool
    err_type: Optional[str] = None
    err_msg: Optional[str] = None
    out_shapes: Optional[List[Tuple[int, ...]]] = None
    out_dtypes: Optional[List[str]] = None


def _try_run(cmd: List[str], timeout_s: int = 5) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout[-4000:],
            "stderr": p.stderr[-4000:],
        }
    except Exception as e:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(e)}


def collect_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "env": {
            k: os.environ.get(k)
            for k in [
                "ASCEND_HOME_PATH",
                "ASCEND_OPP_PATH",
                "PATH",
                "LD_LIBRARY_PATH",
                "DEVICE_ID",
            ]
        },
        "npu_smi": _try_run(["npu-smi", "info"]),
    }
    try:
        import torch  # type: ignore

        info["torch"] = {"version": torch.__version__}
        # Best-effort: force ACLNN path on PTA if supported.
        try:
            if hasattr(torch, "npu") and hasattr(
                torch.npu, "set_compile_mode"
            ):
                torch.npu.set_compile_mode(jit_compile=False)  # type: ignore[attr-defined]
                info["torch"]["set_compile_mode"] = "jit_compile=False"
        except Exception as e:  # noqa: BLE001
            info["torch"]["set_compile_mode_error"] = repr(e)
    except Exception as e:  # noqa: BLE001
        info["torch_import_error"] = repr(e)

    try:
        import torch_npu  # type: ignore

        info["torch_npu"] = {
            "version": getattr(torch_npu, "__version__", None),
        }
        try:
            torch_npu.npu.set_device(  # type: ignore[attr-defined]
                int(os.environ.get("DEVICE_ID", "0"))
            )
            info["torch_npu"]["device_id"] = int(
                os.environ.get("DEVICE_ID", "0")
            )
        except Exception as e:  # noqa: BLE001
            info["torch_npu"]["set_device_error"] = repr(e)
    except Exception as e:  # noqa: BLE001
        info["torch_npu_import_error"] = repr(e)
    return info


def _make_sparse_indices_bsnd(
    *,
    b: int,
    q_s: int,
    n2: int,
    sparse_size: int,
    block_count: int,
    device: str,
) -> "torch.Tensor":
    import torch  # type: ignore

    # Fill with repeating 0..block_count-1 then pad with -1.
    ids = torch.arange(block_count, dtype=torch.int32, device="cpu")
    if sparse_size <= block_count:
        ids = ids[:sparse_size]
    else:
        pad = torch.full(
            (sparse_size - block_count,), -1,
            dtype=torch.int32, device="cpu",
        )
        ids = torch.cat([ids, pad], dim=0)
    ids = ids.view(1, 1, 1, sparse_size).repeat(b, q_s, n2, 1).contiguous()
    return ids.to(device=device)


def _make_sparse_indices_tnd(
    *,
    q_t: int,
    n2: int,
    sparse_size: int,
    block_count: int,
    device: str,
) -> "torch.Tensor":
    import torch  # type: ignore

    ids = torch.arange(block_count, dtype=torch.int32, device="cpu")
    if sparse_size <= block_count:
        ids = ids[:sparse_size]
    else:
        pad = torch.full(
            (sparse_size - block_count,), -1,
            dtype=torch.int32, device="cpu",
        )
        ids = torch.cat([ids, pad], dim=0)
    ids = ids.view(1, 1, sparse_size).repeat(q_t, n2, 1).contiguous()
    return ids.to(device=device)


def run_case(
    *,
    name: str,
    dtype_name: str,
    layout_query: str,
    layout_kv: str,
    sparse_size: int,
    sparse_block_size: int,
    attention_mode: int,
    return_softmax_lse: bool,
    device: str,
    timeout: int = 60,
) -> CaseResult:
    import math

    import torch  # type: ignore
    import torch_npu  # noqa: F401  # type: ignore

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]

    # Keep sizes small but valid per feature doc:
    # - D fixed to 512
    # - N2 fixed to 1
    # - N1 in {1,2,4,8,16,32,64,128}
    b, s1, s2, n1, n2, d = 1, 1, 32, 64, 1, 512
    d_rope = 64
    scale_value = 1.0 / math.sqrt(d)

    cfg: Dict[str, Any] = {
        "dtype": dtype_name,
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "sparse_size": sparse_size,
        "sparse_block_size": sparse_block_size,
        "attention_mode": attention_mode,
        "return_softmax_lse": return_softmax_lse,
        "b": b,
        "s1": s1,
        "s2": s2,
        "n1": n1,
        "n2": n2,
        "d": d,
        "d_rope": d_rope,
    }

    try:
        with case_timeout(timeout):
            if layout_query == "BSND":
                query = torch.randn(
                    (b, s1, n1, d), dtype=dtype, device=device,
                )
            elif layout_query == "TND":
                t1 = b * s1
                query = torch.randn(
                    (t1, n1, d), dtype=dtype, device=device,
                )
            else:
                raise ValueError(
                    f"unsupported layout_query {layout_query}"
                )

            if layout_kv == "BSND":
                key = torch.randn(
                    (b, s2, n2, d), dtype=dtype, device=device,
                )
                value = torch.randn(
                    (b, s2, n2, d), dtype=dtype, device=device,
                )
                block_table = None
                actual_seq_q = torch.tensor(
                    [s1] * b, dtype=torch.int32, device=device,
                )
                actual_seq_kv = torch.tensor(
                    [s2] * b, dtype=torch.int32, device=device,
                )
                block_count = (
                    (s2 + sparse_block_size - 1) // sparse_block_size
                )
                sparse_indices = _make_sparse_indices_bsnd(
                    b=b, q_s=s1, n2=n2,
                    sparse_size=sparse_size,
                    block_count=block_count, device=device,
                )
            elif layout_kv == "TND":
                t2 = b * s2
                key = torch.randn(
                    (t2, n2, d), dtype=dtype, device=device,
                )
                value = torch.randn(
                    (t2, n2, d), dtype=dtype, device=device,
                )
                block_table = None
                # TND requires prefix-sum (cumsum), last == T
                actual_seq_q = torch.tensor(
                    [s1], dtype=torch.int32, device=device,
                ).cumsum(dim=0)
                actual_seq_kv = torch.tensor(
                    [s2], dtype=torch.int32, device=device,
                ).cumsum(dim=0)
                block_count = (
                    (s2 + sparse_block_size - 1) // sparse_block_size
                )
                sparse_indices = _make_sparse_indices_tnd(
                    q_t=query.shape[0], n2=n2,
                    sparse_size=sparse_size,
                    block_count=block_count, device=device,
                )
            elif layout_kv == "PA_BSND":
                # Minimal PA_BSND setup
                block_size = 16
                block_num = 4
                key = torch.randn(
                    (block_num, block_size, n2, d),
                    dtype=dtype, device=device,
                )
                value = torch.randn(
                    (block_num, block_size, n2, d),
                    dtype=dtype, device=device,
                )
                block_table = torch.tensor(
                    [[0, 1, 2, 3]], dtype=torch.int32, device=device,
                )
                actual_seq_q = torch.tensor(
                    [s1] * b, dtype=torch.int32, device=device,
                )
                # For PA_BSND, kv effective length
                kv_eff_len = block_num * block_size
                actual_seq_kv = torch.tensor(
                    [kv_eff_len] * b,
                    dtype=torch.int32, device=device,
                )
                block_count = (
                    (kv_eff_len + sparse_block_size - 1)
                    // sparse_block_size
                )
                sparse_indices = _make_sparse_indices_bsnd(
                    b=b, q_s=s1, n2=n2,
                    sparse_size=sparse_size,
                    block_count=int(block_count),
                    device=device,
                )
            else:
                raise ValueError(
                    f"unsupported layout_kv {layout_kv}"
                )

            if attention_mode == 2:
                query_rope = torch.randn(
                    (*query.shape[:-1], d_rope),
                    dtype=dtype, device=device,
                )
                key_rope = torch.randn(
                    (*key.shape[:-1], d_rope),
                    dtype=dtype, device=device,
                )
            else:
                query_rope = None
                key_rope = None

            # Call PTA API
            out = torch_npu.npu_sparse_flash_attention(  # type: ignore[attr-defined]
                query,
                key,
                value,
                sparse_indices,
                float(scale_value),
                block_table=block_table,
                actual_seq_lengths_query=actual_seq_q,
                actual_seq_lengths_kv=actual_seq_kv,
                query_rope=query_rope,
                key_rope=key_rope,
                sparse_block_size=int(sparse_block_size),
                layout_query=str(layout_query),
                layout_kv=str(layout_kv),
                sparse_mode=3,
                pre_tokens=(2**63 - 1),
                next_tokens=(2**63 - 1),
                attention_mode=int(attention_mode),
                return_softmax_lse=bool(return_softmax_lse),
            )

            outs = list(out) if isinstance(out, (tuple, list)) else [out]
            out_shapes = [tuple(int(x) for x in o.shape) for o in outs]
            out_dtypes = [str(o.dtype) for o in outs]
            return CaseResult(
                name=name, config=cfg, ok=True,
                out_shapes=out_shapes, out_dtypes=out_dtypes,
            )
    except CaseTimeoutError as e:
        return CaseResult(
            name=name, config=cfg, ok=False,
            err_type="CaseTimeoutError", err_msg=str(e),
        )
    except Exception as e:  # noqa: BLE001
        return CaseResult(
            name=name, config=cfg, ok=False,
            err_type=type(e).__name__, err_msg=str(e),
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Probe PTA sparse_flash_attention support range",
    )
    p.add_argument(
        "--device", default="npu:0", help="Device string, e.g. npu:0",
    )
    p.add_argument(
        "--out",
        default="pta_sparse_flash_attention_probe.json",
        help="Output JSON filename",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick mode: only run core parameter combinations",
    )
    p.add_argument(
        "--timeout", type=int, default=60,
        help="Per-case timeout in seconds (default: 60, Unix only)",
    )
    args = p.parse_args()

    env_info = collect_env_info()

    # Test matrix
    if args.quick:
        # Quick mode: reduced matrix for fast verification
        dtypes = ["float16", "bfloat16"]
        layouts = [("BSND", "BSND"), ("TND", "TND")]
        sparse_sizes = [2048]
        sparse_block_sizes = [4]
        attention_modes = [0]
        return_lse_flags = [False]
    else:
        # Full matrix
        dtypes = ["float16", "bfloat16"]
        layouts = [("BSND", "BSND"), ("TND", "TND"), ("BSND", "PA_BSND")]
        sparse_sizes = [16, 128, 2048]
        sparse_block_sizes = [1, 2, 4, 8]
        attention_modes = [0, 2]
        return_lse_flags = [False, True]

    results: List[CaseResult] = []
    total = (
        len(dtypes) * len(layouts) * len(sparse_sizes)
        * len(sparse_block_sizes) * len(attention_modes)
        * len(return_lse_flags)
    )
    idx = 0
    for dtype_name in dtypes:
        for layout_query, layout_kv in layouts:
            for sparse_size in sparse_sizes:
                for sparse_block_size in sparse_block_sizes:
                    for attention_mode in attention_modes:
                        for return_lse in return_lse_flags:
                            idx += 1
                            name = (
                                f"dtype={dtype_name},"
                                f"lq={layout_query},"
                                f"lkv={layout_kv},"
                                f"sparse={sparse_size},"
                                f"blk={sparse_block_size},"
                                f"am={attention_mode},"
                                f"lse={return_lse}"
                            )
                            print(
                                f"[{idx}/{total}] {name}",
                                flush=True,
                            )
                            results.append(
                                run_case(
                                    name=name,
                                    dtype_name=dtype_name,
                                    layout_query=layout_query,
                                    layout_kv=layout_kv,
                                    sparse_size=sparse_size,
                                    sparse_block_size=sparse_block_size,
                                    attention_mode=attention_mode,
                                    return_softmax_lse=return_lse,
                                    device=args.device,
                                    timeout=args.timeout,
                                )
                            )

    payload = {
        "env_info": env_info,
        "results": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r.ok),
            "fail": sum(1 for r in results if not r.ok),
            "timeout": sum(
                1 for r in results
                if not r.ok and r.err_type == "CaseTimeoutError"
            ),
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
