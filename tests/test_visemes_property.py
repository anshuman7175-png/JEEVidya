"""
Property-based laws of the viseme engine (Terminal Plan Phase 0.2).
═══════════════════════════════════════════════════════════════════════
tests/test_visemes.py pins *examples*; this module pins *laws*. Hypothesis
explores the input space adversarially — conjunct pile-ups, matra-only
tokens, degenerate durations — and shrinks every failure to a minimal
counterexample, which is exactly the coverage a 30-word fixture cannot
give. Every law here is quoted from the plan's verification matrix:

  C1  g2p is total, enum-closed, jaw-mapped, and never emits repeats
  C5  a *written* matra is never deleted; every consonant word keeps
      at least one vowel; word boundaries compose
  §7.4 coalescing preserves span, contiguity, and the min-duration bound
  C3  weight mass is conserved at every instant, on every track

If hypothesis is not installed the conftest dependency gate skips this
module loudly (never a silent pass) — the pure-logic gate stays runnable
on a bare interpreter.
"""
from __future__ import annotations

from itertools import pairwise

from hypothesis import given, settings, strategies as st

from engine.visemes import (JAW, SUBFRAMES, VOWELS, V, VisemeEvent,
                            VisemeTrack, coalesce_events, g2p,
                            visemes_for_word)

# ═══════════════════════════════════════════
# strategies
# ═══════════════════════════════════════════

_DEV_CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
_DEV_MATRAS = "ािीुूेैोौ"
_VIRAMA = "्"
_DEV_INDEPENDENT = "अआइईउऊएऐओऔ"

# Full Devanagari block + Latin + digits + the symbols turn (j) exercises:
# whatever the scriptwriter emits, g2p must be total over it.
_ANY_CHAR = st.one_of(
    st.characters(min_codepoint=0x0900, max_codepoint=0x097F),
    st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    st.sampled_from("²³θπ√°—…‘’“”"),
)
any_text = st.text(alphabet=_ANY_CHAR, min_size=0, max_size=24)

# A pronounceable Devanagari word: consonants, optional matras/virama.
_dev_unit = st.one_of(
    st.sampled_from(_DEV_CONSONANTS),
    st.sampled_from(_DEV_CONSONANTS).flatmap(
        lambda c: st.sampled_from(_DEV_MATRAS).map(lambda m: c + m)),
    st.sampled_from(_DEV_CONSONANTS).map(lambda c: c + _VIRAMA),
    st.sampled_from(_DEV_INDEPENDENT),
)
dev_word = st.lists(_dev_unit, min_size=1, max_size=6).map("".join)


def _dedup(seq):
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


# ═══════════════════════════════════════════
# C1 — totality and enum closure
# ═══════════════════════════════════════════

@settings(max_examples=500)
@given(any_text)
def test_g2p_is_total_enum_closed_and_jaw_mapped(text):
    out = g2p(text)
    for v in out:
        assert isinstance(v, V)
        assert v in JAW
    # the engine's own postcondition: adjacent duplicates are collapsed
    assert out == _dedup(out)


@settings(max_examples=300)
@given(any_text)
def test_g2p_is_deterministic(text):
    assert g2p(text) == g2p(text)


@settings(max_examples=300)
@given(st.text(alphabet=st.sampled_from(" \t!?.,;:-—()[]{}\"'"),
               max_size=12))
def test_unmappable_tokens_yield_usable_rest(junk):
    """Callers index the result unconditionally: silence, punctuation and
    whitespace must produce REST, never an empty list."""
    assert visemes_for_word(junk) == [V.REST.value]


# ═══════════════════════════════════════════
# C5 — schwa deletion never overreaches
# ═══════════════════════════════════════════

def _licenses_a_vowel(word: str) -> bool:
    """True iff the orthography itself promises a vowel: a matra, an
    independent vowel, or a bare (non-halanted) consonant carrying the
    inherent schwa. A fully halanted word like क् is the author saying
    "no vowel" — Hypothesis found that counterexample on its first run,
    and the engine's behaviour there is correct, not a bug."""
    chars = list(word)
    for i, ch in enumerate(chars):
        if ch in _DEV_MATRAS or ch in _DEV_INDEPENDENT:
            return True
        if ch in _DEV_CONSONANTS and (i + 1 >= len(chars)
                                      or chars[i + 1] != _VIRAMA):
            return True
    return False


@settings(max_examples=500)
@given(dev_word)
def test_no_word_is_stripped_of_every_vowel(word):
    """Deletion targets only *inserted* schwas, and never a word's last
    vowel: whenever the orthography licenses a vowel, one survives."""
    out = g2p(word)
    if _licenses_a_vowel(word) and any(v is not V.REST for v in out):
        assert any(v in VOWELS for v in out), word


@settings(max_examples=500)
@given(st.lists(st.sampled_from(_DEV_CONSONANTS), min_size=1, max_size=4),
       st.sampled_from(_DEV_MATRAS))
def test_written_matra_always_survives(consonants, matra):
    """The hard C5 invariant: a vowel the author wrote is untouchable.
    Place one matra anywhere in a consonant word — the output must keep
    a vowel from it."""
    word = consonants[0] + matra + "".join(consonants[1:])
    assert any(v in VOWELS for v in g2p(word)), word


@settings(max_examples=300)
@given(dev_word, dev_word)
def test_word_boundaries_compose(a, b):
    """g2p over "a b" must equal g2p(a) ++ g2p(b) up to the engine's own
    adjacent-repeat collapse — whitespace is a pure anchor, never a
    semantic input to either word's schwa deletion."""
    assert g2p(f"{a} {b}") == _dedup(g2p(a) + g2p(b))


# ═══════════════════════════════════════════
# §7.4 — coalescing conservation laws
# ═══════════════════════════════════════════

_contiguous_events = st.lists(
    st.tuples(st.sampled_from(list(V)),
              st.floats(min_value=1.0, max_value=60.0,
                        allow_nan=False, allow_infinity=False)),
    min_size=1, max_size=10,
).map(lambda pairs: [
    VisemeEvent(v, sum(d for _, d in pairs[:i]),
                sum(d for _, d in pairs[:i + 1]))
    for i, (v, _) in enumerate(pairs)
])


@settings(max_examples=400)
@given(_contiguous_events,
       st.floats(min_value=1.0, max_value=25.0,
                 allow_nan=False, allow_infinity=False))
def test_coalesce_conserves_span_contiguity_and_min_duration(evs, min_dur):
    first, last = evs[0].start_ms, evs[-1].end_ms
    out = coalesce_events(evs, min_dur)
    assert out, "coalescing must never empty a non-empty timeline"
    assert abs(out[0].start_ms - first) < 1e-6
    assert abs(out[-1].end_ms - last) < 1e-6
    for a, b in pairwise(out):
        assert abs(b.start_ms - a.end_ms) < 1e-6, "gap opened by merge"
    # every survivor meets the floor — or is the irreducible last event
    assert len(out) == 1 or all(e.dur >= min_dur - 1e-6 for e in out)


# ═══════════════════════════════════════════
# C3 — weight mass conservation on arbitrary tracks
# ═══════════════════════════════════════════

class _Word:
    def __init__(self, text: str, start_ms: float, end_ms: float) -> None:
        self.text = text
        self.start_ms = start_ms
        self.end_ms = end_ms


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(dev_word,
                          st.floats(min_value=60.0, max_value=600.0)),
                min_size=1, max_size=4),
       st.floats(min_value=0.0, max_value=1.0))
def test_weight_mass_is_conserved_everywhere(word_specs, frac):
    """At any instant — inside speech, in a cross-fade, in a release
    tail, in dead silence — the blend weights are a distribution: they
    sum to one and are non-negative, and jaw stays in [0, 1]."""
    t, words = 0.0, []
    for text, dur in word_specs:
        words.append(_Word(text, t, t + dur))
        t += dur + 90.0
    tr = VisemeTrack.from_words(words, t + 500.0)
    probe = frac * (t + 500.0)
    for t_ms in (probe, probe + 3.7):
        weights, jaw = tr.weights_at(t_ms)
        assert abs(sum(weights.values()) - 1.0) < 1e-6, t_ms
        assert all(w >= -1e-9 for w in weights.values())
        assert -1e-9 <= jaw <= 1.0 + 1e-9
        fw, fj = tr.weights_at_frame(t_ms, fps=30)
        assert abs(sum(fw.values()) - 1.0) < 1e-6, t_ms
        assert -1e-9 <= fj <= 1.0 + 1e-9
    assert SUBFRAMES >= 2
