#!/usr/bin/env python3
"""Lightweight self-checks for the dtype probe framework and scaffold template.

This script intentionally stays dependency-light:
- stdlib only
- no real MindSpore runtime required
- no torch/torch_npu execution required
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_PATH = SCRIPT_DIR / "dtype_probe_execution_framework.py"
TEMPLATE_PATH = SCRIPT_DIR.parent / "template" / "dtype_probe_operator_scaffold.template.py"


def _install_fake_mindspore() -> None:
    if "mindspore" in sys.modules and "mindspore.ops" in sys.modules:
        return

    ms = types.ModuleType("mindspore")
    ms_ops = types.ModuleType("mindspore.ops")

    class FakeTensor:
        pass

    ms.Tensor = FakeTensor
    for dtype_name in (
        "bool_",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ):
        setattr(ms, dtype_name, dtype_name)

    sys.modules["mindspore"] = ms
    sys.modules["mindspore.ops"] = ms_ops


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTemplateFramework:
    BACKWARD_NOT_APPLICABLE = object()
    DEFAULT_DTYPE_NAMES = ("float16",)

    @staticmethod
    def make_unary_sample(dtype, **kwargs):
        return {"dtype": dtype, **kwargs}

    class OperatorProbeSpec:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


class DtypeProbeFrameworkSelfcheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_fake_mindspore()
        cls.framework = _load_module("dtype_probe_framework_selfcheck_fw", FRAMEWORK_PATH)
        sys.modules["dtype_probe_execution_framework"] = cls.framework
        cls.template = _load_module("dtype_probe_framework_selfcheck_template", TEMPLATE_PATH)

    def test_template_build_samples_accepts_injected_framework(self):
        samples = self.template.build_samples("float16", "float16", framework=_FakeTemplateFramework)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["dtype"], "float16")
        self.assertEqual(samples[0]["sample_name"], "float16_basic")

    def test_template_build_operator_probes_uses_injected_framework(self):
        specs = self.template.build_operator_probes(_FakeTemplateFramework)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].candidate_dtypes, _FakeTemplateFramework.DEFAULT_DTYPE_NAMES)

    def test_backward_not_applicable_is_reported_explicitly(self):
        sample = self.framework.ProbeSample(op_input=None, op_args=(), op_kwargs={}, sample_name="no_grad", grad_position=())
        spec = self.framework.OperatorProbeSpec(
            op_name="fake_no_grad",
            build_samples=lambda dtype_name, ms_dtype: [sample],
            forward_runner=lambda _: None,
            backward_runner=lambda _: self.framework.BACKWARD_NOT_APPLICABLE,
            candidate_dtypes=("float16",),
            declared_doc_forward_dtypes=("float16",),
            declared_doc_backward_dtypes=(),
            supports_backward=True,
        )
        summary = self.framework.execute_operator_probe(spec)
        self.assertEqual(summary["backward_not_applicable_dtypes"], ["float16"])
        self.assertEqual(summary["backward_supported_dtypes"], [])

    def test_aggregate_direction_status_ignores_not_applicable_when_supported_exists(self):
        result = self.framework._aggregate_direction_status(
            [{"status": "supported"}, {"status": "not_applicable"}],
            supports_backward=True,
        )
        self.assertEqual(result, "supported")

    def test_classifier_keeps_generic_invalid_dtype_as_issue(self):
        spec = types.SimpleNamespace(
            unsupported_error_patterns=(),
            unsupported_error_keywords=(),
            error_classifier=None,
        )
        issue = self.framework.classify_failure(
            RuntimeError("input dtype is invalid because sample wiring is wrong"),
            spec=spec,
            direction="forward",
            dtype_name="float64",
            sample=None,
        )
        unsupported = self.framework.classify_failure(
            RuntimeError("Tensor gradOutput not implemented for DT_DOUBLE, should be in dtype support list [DT_FLOAT]"),
            spec=spec,
            direction="backward",
            dtype_name="float64",
            sample=None,
        )
        self.assertEqual(issue, "sample_or_function_issue")
        self.assertEqual(unsupported, "unsupported_dtype")

    def test_doc_derived_summary_helper_uses_unified_schema(self):
        summary = self.framework.build_doc_derived_operator_summary(
            op_name="fake_doc_only",
            declared_doc_forward_dtypes=("float16", "float32"),
            declared_doc_backward_dtypes=("float16",),
            notes="doc-only fallback",
        )
        self.assertEqual(summary["probe_backend"], "doc")
        self.assertEqual(summary["dtype_declaration_source"], "doc_derived")
        self.assertTrue(summary["runtime_probe_skipped"])
        self.assertEqual(summary["forward_supported_dtypes"], ["float16", "float32"])
        self.assertEqual(summary["backward_supported_dtypes"], ["float16"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
