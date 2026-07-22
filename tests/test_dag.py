"""Tier 0 — jvmake DAG: incrementality, resume-after-crash, cycle safety."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cache import BuildCache, key_of
from pipeline.dag import Graph


def _writer(tmp_path, name, content, calls):
    """A build fn that records it ran and produces a file."""
    def build(node):
        calls.append(name)
        p = tmp_path / f"{name}.txt"
        p.write_text(content)
        return str(p)
    return build


def _two_node_graph(tmp_path, cache, content_a, calls):
    """a → b, where b's key depends on a's key (input propagation)."""
    g = Graph(cache)
    a = g.node("a", key_of("a", content_a), "txt",
               _writer(tmp_path, "a", content_a, calls))
    b = g.node("b", key_of("b", a.key), "txt",
               _writer(tmp_path, "b", content_a + "!", calls), deps=[a])
    return g, b


def test_cold_build_runs_everything(tmp_path):
    cache = BuildCache(root=str(tmp_path / "c"))
    calls = []
    g, b = _two_node_graph(tmp_path, cache, "v1", calls)
    path = g.build(b)
    assert calls == ["a", "b"]
    assert open(path).read() == "v1!"
    assert g.summary(b).startswith("0 cached · 2 built")


def test_warm_build_runs_nothing(tmp_path):
    cache = BuildCache(root=str(tmp_path / "c"))
    g1, b1 = _two_node_graph(tmp_path, cache, "v1", [])
    g1.build(b1)

    calls = []
    g2, b2 = _two_node_graph(tmp_path, cache, "v1", calls)
    path = g2.build(b2)
    assert calls == []                      # 100% cache hits
    assert open(path).read() == "v1!"
    assert g2.summary(b2).startswith("2 cached · 0 built")


def test_changed_input_invalidates_downstream(tmp_path):
    cache = BuildCache(root=str(tmp_path / "c"))
    g1, b1 = _two_node_graph(tmp_path, cache, "v1", [])
    g1.build(b1)

    calls = []
    g2, b2 = _two_node_graph(tmp_path, cache, "v2", calls)
    g2.build(b2)
    assert calls == ["a", "b"]              # b's key embeds a's key


def test_only_changed_branch_rebuilds(tmp_path):
    """The Tier 0 promise: edit one line → one branch rebuilds."""
    cache = BuildCache(root=str(tmp_path / "c"))

    def make_graph(content_b, calls):
        g = Graph(cache)
        a = g.node("a", key_of("a", "same"), "txt",
                   _writer(tmp_path, "a", "same", calls))
        b = g.node("b", key_of("b", content_b), "txt",
                   _writer(tmp_path, "b", content_b, calls))
        top = g.node("top", key_of("top", a.key, b.key), "txt",
                     _writer(tmp_path, "top", "t", calls), deps=[a, b])
        return g, top

    g1, t1 = make_graph("v1", [])
    g1.build(t1)

    calls = []
    g2, t2 = make_graph("v2", calls)        # only b changed
    g2.build(t2)
    assert calls == ["b", "top"]            # a stayed a cache hit


def test_crash_resume(tmp_path):
    """Crash mid-build → next run resumes exactly where it died."""
    cache = BuildCache(root=str(tmp_path / "c"))
    calls = []

    g = Graph(cache)
    a = g.node("a", key_of("a"), "txt", _writer(tmp_path, "a", "a", calls))

    def boom(node):
        raise RuntimeError("simulated render crash")
    b = g.node("b", key_of("b", a.key), "txt", boom, deps=[a])

    with pytest.raises(RuntimeError):
        g.build(b)
    assert calls == ["a"]                   # a finished before the crash

    calls.clear()
    g2 = Graph(cache)
    a2 = g2.node("a", key_of("a"), "txt", _writer(tmp_path, "a", "a", calls))
    b2 = g2.node("b", key_of("b", a2.key), "txt",
                 _writer(tmp_path, "b", "b", calls), deps=[a2])
    g2.build(b2)
    assert calls == ["b"]                   # resume: a is already cached


def test_force_rebuilds_everything(tmp_path):
    cache = BuildCache(root=str(tmp_path / "c"))
    g1, b1 = _two_node_graph(tmp_path, cache, "v1", [])
    g1.build(b1)

    calls = []
    g2, b2 = _two_node_graph(tmp_path, cache, "v1", calls)
    g2.build(b2, force=True)
    assert calls == ["a", "b"]


def test_status_is_a_dry_run(tmp_path):
    cache = BuildCache(root=str(tmp_path / "c"))
    calls = []
    g, b = _two_node_graph(tmp_path, cache, "v1", calls)
    statuses = g.status(b)
    assert [(n.name, hit) for n, hit in statuses] == [("a", False), ("b", False)]
    assert calls == []                      # status never builds


def test_cycle_detection(tmp_path):
    g = Graph(BuildCache(root=str(tmp_path)))
    a = g.node("a", key_of("a"), "txt", lambda n: "")
    b = g.node("b", key_of("b"), "txt", lambda n: "", deps=[a])
    a.deps.append(b)                        # a ↔ b
    with pytest.raises(ValueError):
        g.build(b)


def test_duplicate_node_names_rejected(tmp_path):
    g = Graph(BuildCache(root=str(tmp_path)))
    g.node("a", key_of("a"), "txt", lambda n: "")
    with pytest.raises(ValueError):
        g.node("a", key_of("a2"), "txt", lambda n: "")


def test_progress_callback_reports_every_node(tmp_path):
    cache = BuildCache(root=str(tmp_path / "c"))
    g, b = _two_node_graph(tmp_path, cache, "v1", [])
    seen = []
    g.build(b, on_progress=lambda done, total, node: seen.append(
        (done, total, node.name, node.cached)))
    assert seen == [(1, 2, "a", False), (2, 2, "b", False)]
