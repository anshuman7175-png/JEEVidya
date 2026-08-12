"""
Viseme engine — the pure-logic constitution (Terminal Plan Track C).
═══════════════════════════════════════════════════════════════════════
This module is the crown of the pipeline and the cheapest thing in it to
test: no numpy, no PIL, no ffmpeg, no audio. Every assertion here runs in
milliseconds on any machine, which is exactly why the plan puts the
pure-function suite in CI ahead of every render-based gate.

Coverage maps 1:1 onto the plan's verification matrix:

  C1  taxonomy      — enum closure, matras, virama, conjuncts, loanwords,
                      digits, and randomized (Hypothesis-style) fuzzing
  C3  coarticulation— weight mass conservation, boundary blending,
                      bilabial closure at plosive midpoints
  C4  envelope      — attack/release, REST after silence
  C5  schwa         — Ohala deletion, and the invariant that a *written*
                      matra can never be deleted
  §7.4 sub-frame    — box-filtered sampling conserves weight mass
"""
from __future__ import annotations

import random

from engine.visemes import (JAW, ROUNDED, SUBFRAMES, VOWELS, V,
                            AmplitudeEnvelope, VisemeEvent, VisemeTrack,
                            coalesce_events, g2p, visemes_for_word)

# ═══════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════

_MATRAS = "\u093e\u093f\u0940\u0941\u0942\u0947\u0948\u094b\u094c"
_CONSONANTS = "\u0915\u0916\u0917\u091c\u0924\u0926\u0928\u092a\u092b\u092c\u092e\u0932\u0935\u0938\u0939"
_VIRAMA = "\u094d"


class _Word:
    """Minimal stand-in for a VTT word event (text + span in ms)."""

    def __init__(self, text: str, start_ms: float, end_ms: float) -> None:
        self.text = text
        self.start_ms = start_ms
        self.end_ms = end_ms


def _track(*spans):
    """Build a VisemeTrack from (viseme, start, end) triples."""
    evs = [VisemeEvent(v, s, e) for v, s, e in spans]
    return VisemeTrack(evs, evs[-1].end_ms + 100.0)


# ═══════════════════════════════════════════
# C5 — Ohala schwa deletion
# ═══════════════════════════════════════════
#
# Devanagari writes the inherent schwa that modern Hindi does not
# pronounce. Retaining it is the most visible lip-sync error available,
# because OPEN_A has the largest jaw drop in the inventory (JAW == 1.0):
# the character would end nearly every Hindi word with a wide-open mouth.

WORD_FINAL_CASES = {
    # word      romanization   expected viseme run
    "\u0915\u092e\u0932": (            # कमल  kamal
        [V.RETROFLEX, V.OPEN_A, V.BILABIAL, V.OPEN_A, V.DENTAL]),
    "\u0928\u092e\u0915": (            # नमक  namak
        [V.DENTAL, V.OPEN_A, V.BILABIAL, V.OPEN_A, V.RETROFLEX]),
    "\u092c\u0932": (                  # बल   bal
        [V.BILABIAL, V.OPEN_A, V.DENTAL]),
    "\u0935\u0947\u0917": (            # वेग  veg
        [V.LABIODENTAL, V.MID_E, V.RETROFLEX]),
}


def test_word_final_schwa_is_deleted():
    for word, expected in WORD_FINAL_CASES.items():
        assert g2p(word) == expected, f"{word}: {g2p(word)}"


def test_word_final_schwa_leaves_no_trailing_open_a():
    """The regression that matters visually: no wide-open jaw parked on
    the end of a consonant-final Hindi word."""
    for word in WORD_FINAL_CASES:
        assert g2p(word)[-1] is not V.OPEN_A, word


def test_medial_schwa_deletes_in_vc_cv():
    # नमकीन  namkeen — the medial schwa in V C _ C V drops, the first
    # schwa stays (deletion is right-to-left and non-adjacent).
    word = "\u0928\u092e\u0915\u0940\u0928"
    assert g2p(word) == [V.DENTAL, V.OPEN_A, V.BILABIAL,
                         V.RETROFLEX, V.CLOSED_I, V.DENTAL]


def test_monosyllable_keeps_its_only_vowel():
    """न is "na", not a vowel-less consonant: deletion must never strip a
    word's last remaining vowel."""
    assert g2p("\u0928") == [V.DENTAL, V.OPEN_A]
    for cons in _CONSONANTS:
        out = g2p(cons)
        assert any(v in VOWELS for v in out), cons


def test_written_matra_is_never_deleted():
    """The hard invariant: only schwas the G2P *inserted* are eligible for
    deletion. A vowel the author actually wrote always survives."""
    rng = random.Random(20240612)
    for _ in range(2000):
        word = "".join(rng.choice(_CONSONANTS + _MATRAS)
                       for _ in range(rng.randint(1, 8)))
        if not any(c in _MATRAS for c in word):
            continue
        out = g2p(word)
        assert any(v in VOWELS for v in out), word


def test_conjunct_virama_suppresses_schwa():
    # क्ष — virama binds the cluster, so no schwa between क and ष
    assert g2p("\u0915" + _VIRAMA + "\u0937") == [V.RETROFLEX, V.DENTAL,
                                                  V.OPEN_A]
    # त्वरण  tvaran — leading cluster, medial schwa retained, final dropped
    assert g2p("\u0924" + _VIRAMA + "\u0935\u0930\u0923") == [
        V.DENTAL, V.LABIODENTAL, V.OPEN_A, V.RETROFLEX, V.OPEN_A,
        V.RETROFLEX]


def test_multiword_text_anchors_each_word_boundary():
    """Word-final deletion needs a word boundary; a whitespace-joined
    string must behave exactly like its tokens concatenated."""
    # कमल बल — chosen so the tokens do not abut on the same viseme
    # class (कमल ends DENTAL, बल opens BILABIAL), because adjacent
    # duplicates are deliberately collapsed into one sustained shape.
    a, b = "\u0915\u092e\u0932", "\u092c\u0932"
    joined = g2p(f"{a} {b}")
    assert joined == g2p(a) + g2p(b)
    assert joined.count(V.OPEN_A) == 3


# ═══════════════════════════════════════════
# C1 — taxonomy closure & robustness
# ═══════════════════════════════════════════

def test_g2p_never_emits_out_of_enum():
    for word in list(WORD_FINAL_CASES) + ["velocity", "9.8", "sin\u00b2\u03b8"]:
        for v in g2p(word):
            assert isinstance(v, V)
            assert v in JAW


def test_g2p_fuzz_never_crashes():
    """Property-based coverage per the plan: random Devanagari / Latin /
    mixed strings must never crash, never leave the enum, and never
    produce a viseme without a jaw value."""
    rng = random.Random(99)
    alphabet = ([chr(c) for c in range(0x0900, 0x0980)]
                + list("abcdefghijklmnopqrstuvwxyz0123456789 .,?!\u00b2")
                + [_VIRAMA, "\u093c", "\u0902"])
    for _ in range(4000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 16)))
        out = g2p(s)
        assert all(isinstance(v, V) and v in JAW for v in out)


def test_visemes_for_word_is_never_empty():
    """Callers index the result unconditionally, so an unmappable token
    must still yield a usable REST rather than an IndexError."""
    for token in ["", "!!!", "\u0964", "---", " "]:
        assert visemes_for_word(token) == [V.REST.value]


def test_digits_are_spoken_not_silent():
    """A numeric token must move the mouth: "9.8" is read aloud."""
    out = g2p("9")
    assert out and V.REST not in out


def test_hinglish_code_switch_survives():
    out = g2p("iss equation mein velocity substitute karo")
    assert len(out) > 20
    assert all(v in JAW for v in out)


# ═══════════════════════════════════════════
# C3 — coarticulation & weight mass
# ═══════════════════════════════════════════

def test_weights_always_sum_to_one():
    tr = VisemeTrack.from_words(
        [_Word("\u0915\u092e\u0932", 0, 400),
         _Word("\u0928\u092e\u0915", 520, 900)], 1200.0)
    t = 0.0
    while t <= tr.turn_end_ms:
        weights, jaw = tr.weights_at(t)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"t={t} sum={total}"
        assert all(w >= -1e-9 for w in weights.values())
        assert 0.0 - 1e-9 <= jaw <= 1.0 + 1e-9
        t += 5.0


def test_subframe_sampling_conserves_weight_mass():
    """§7.4: box-filtering across SUBFRAMES points must average
    normalized distributions into a normalized distribution."""
    assert SUBFRAMES >= 2
    tr = _track((V.OPEN_A, 0, 120), (V.BILABIAL, 120, 150),
                (V.MID_E, 150, 300))
    for t in range(0, 320, 3):
        weights, _ = tr.weights_at_frame(float(t), fps=30)
        assert abs(sum(weights.values()) - 1.0) < 1e-6, t


def test_bilabial_reaches_full_closure_at_midpoint():
    """C3 acceptance, as a unit test rather than an eyeball: a plosive
    must actually close the lips, not sit half-open."""
    tr = _track((V.OPEN_A, 0, 200), (V.BILABIAL, 200, 300),
                (V.OPEN_A, 300, 500))
    weights, jaw = tr.weights_at(250.0)
    assert weights.get(V.BILABIAL, 0.0) > 0.9, weights
    assert jaw < 0.2, jaw


def test_blend_is_symmetric_across_a_boundary():
    """No mouth pop: the mix must pass through the boundary continuously
    instead of snapping to 100% of the incoming viseme."""
    tr = _track((V.OPEN_A, 0, 200), (V.MID_E, 200, 400))
    before, _ = tr.weights_at(199.0)
    after, _ = tr.weights_at(201.0)
    assert before.get(V.MID_E, 0.0) > 0.0, "no anticipation before boundary"
    assert after.get(V.OPEN_A, 0.0) > 0.0, "incoming half-blend missing"
    assert abs(before.get(V.MID_E, 0.0) - after.get(V.MID_E, 0.0)) < 0.25


def test_anticipatory_rounding_before_rounded_vowel():
    tr = _track((V.OPEN_A, 0, 200), (V.ROUNDED_TENSE, 200, 400))
    weights, _ = tr.weights_at(175.0)
    assert any(v in ROUNDED for v in weights), weights


# ═══════════════════════════════════════════
# C4 / C8 — envelope, silence, breath
# ═══════════════════════════════════════════

def test_mouth_closes_after_silence():
    tr = _track((V.OPEN_A, 0, 100))
    weights, jaw = tr.weights_at(100.0 + 400.0)
    assert weights.get(V.REST, 0.0) > 0.99
    assert jaw < 1e-6


def test_long_pause_reports_a_breath():
    tr = _track((V.OPEN_A, 0, 200), (V.MID_E, 1400, 1600))
    assert tr.breath_pauses(), "a 1.2 s gap must trigger an inhale"


def test_amplitude_envelope_attacks_faster_than_it_releases():
    env = AmplitudeEnvelope(fps=30)
    for _ in range(3):
        env.step(0.0)
    peak = env.level
    for _ in range(3):
        env.step(-50.0)
    assert peak > 0.3
    assert env.level > 0.0, "release must not slam shut in 3 frames"


# ═══════════════════════════════════════════
# §7.4 — coalescing
# ═══════════════════════════════════════════

def test_coalesce_preserves_span_and_contiguity():
    rng = random.Random(4242)
    for _ in range(300):
        t = 0.0
        evs = []
        for _ in range(rng.randint(1, 9)):
            d = rng.uniform(1.0, 40.0)
            evs.append(VisemeEvent(rng.choice(list(V)), t, t + d))
            t += d
        first, last = evs[0].start_ms, evs[-1].end_ms
        out = coalesce_events(evs, 12.0)
        assert out
        assert abs(out[0].start_ms - first) < 1e-6
        assert abs(out[-1].end_ms - last) < 1e-6
        for a, b in zip(out, out[1:]):
            assert abs(b.start_ms - a.end_ms) < 1e-6


def test_short_bilabial_absorbs_rather_than_vanishing():
    """A 4 ms plosive closure is articulatorily undeletable: it must eat
    a neighbour's time, never be merged away into an open vowel."""
    evs = [VisemeEvent(V.OPEN_A, 0, 100),
           VisemeEvent(V.BILABIAL, 100, 104),
           VisemeEvent(V.OPEN_A, 104, 200)]
    out = coalesce_events(evs, 12.0)
    assert V.BILABIAL in [e.viseme for e in out]


def test_from_words_keeps_events_inside_their_word_span():
    words = [_Word("\u0915\u092e\u0932", 0, 300),
             _Word("\u0928\u092e\u0915", 400, 700)]
    tr = VisemeTrack.from_words(words, 800.0)
    assert tr.events
    for e in tr.events:
        assert e.end_ms >= e.start_ms
        assert any(w.start_ms - 1e-6 <= e.start_ms and e.end_ms <= w.end_ms + 1e-6
                   for w in words), e
