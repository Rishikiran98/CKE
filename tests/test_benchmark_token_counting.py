"""One counter, every arm.

A token figure must not differ between arms because of how it was counted.
The three pipelines used to build a counter apiece; they now take one, and
cannot be constructed without it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cke.evaluation.token_counter import TokenCounter

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_cke_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_cke_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_module()

PIPELINES = ("RAGPipeline", "CKELitePipeline", "HybridPipeline")


@pytest.mark.parametrize("name", PIPELINES)
def test_a_pipeline_cannot_be_built_without_a_counter(name):
    with pytest.raises(TypeError):
        getattr(bench, name)(strict=False)


def test_every_arm_counts_with_the_same_object():
    counter = TokenCounter(strict=True)
    built = [
        getattr(bench, name)(token_counter=counter, strict=False) for name in PIPELINES
    ]

    assert all(pipeline._counter is counter for pipeline in built)


def test_the_script_no_longer_defines_a_word_count_estimator():
    """The multiplier is gone from the driver, not merely unused.

    It existed twice — as a class and as a module-level helper — and a copy
    left behind is a copy that gets called again.
    """
    source = (ROOT / "scripts" / "run_cke_benchmark.py").read_text(encoding="utf-8")

    assert "WORDS_TO_TOKENS" not in source
    assert "_token_count_static" not in source
    assert "len(text.split()) * 1.3" not in source


def test_the_summary_names_the_counter_that_produced_its_figures():
    counter = TokenCounter(strict=True)
    summary = bench.produce_summary(
        {"rag_k10": {"median_tokens": 100.0}, "cke_n12": {"median_tokens": 10.0}},
        counter,
    )

    assert summary["prompt_token_counter"] == counter.description
    assert summary["prompt_token_figures_are_estimates"] is False
