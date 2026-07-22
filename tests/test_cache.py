"""Tier 0 — BuildCache: the memory of the factory must never lie."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cache import BuildCache, key_of


def test_key_of_separates_parts():
    # ("ab","c") must never collide with ("a","bc")
    assert key_of("ab", "c") != key_of("a", "bc")
    assert key_of("x") != key_of("x", "")
    assert key_of(1, 2) == key_of("1", "2")  # stable stringification


def test_key_of_is_deterministic():
    assert key_of("seg", 1080, 1920, 30) == key_of("seg", 1080, 1920, 30)


def test_put_get_fetch(tmp_path):
    cache = BuildCache(root=str(tmp_path / "cache"))
    src = tmp_path / "a.txt"
    src.write_text("hello")

    key = key_of("artifact")
    assert cache.get(key, "txt") is None            # cold miss

    stored = cache.put(key, "txt", str(src))
    assert stored is not None
    with open(stored) as f:
        assert f.read() == "hello"
    assert cache.get(key, "txt") == stored          # warm hit

    dst = tmp_path / "out" / "b.txt"
    assert cache.fetch(key, "txt", str(dst))
    assert dst.read_text() == "hello"


def test_put_missing_source_is_none(tmp_path):
    cache = BuildCache(root=str(tmp_path))
    assert cache.put(key_of("x"), "txt", str(tmp_path / "nope.txt")) is None


def test_empty_artifact_is_a_miss(tmp_path):
    # Zero-byte files (half-written crashes) must never count as hits
    cache = BuildCache(root=str(tmp_path))
    src = tmp_path / "empty.bin"
    src.write_bytes(b"")
    cache.put(key_of("e"), "bin", str(src))
    assert cache.get(key_of("e"), "bin") is None


def test_text_sidecar(tmp_path):
    cache = BuildCache(root=str(tmp_path))
    key = key_of("duration")
    assert cache.get_text(key, "ms") is None
    cache.put_text(key, "ms", "4231")
    assert cache.get_text(key, "ms") == "4231"


def test_clear_and_stats(tmp_path):
    cache = BuildCache(root=str(tmp_path / "c"))
    src = tmp_path / "f.txt"
    src.write_text("data")
    cache.put(key_of("s"), "txt", str(src))
    assert cache.stats()["files"] >= 1
    cache.clear()
    assert cache.stats()["files"] == 0
    assert cache.get(key_of("s"), "txt") is None
