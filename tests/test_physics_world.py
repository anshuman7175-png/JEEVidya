"""Tier 2 — Physics world: determinism, reactions, constraints."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.physics_world import PhysicsWorld


def _world(seed=42):
    w = PhysicsWorld(1080, 1920, seed=seed)
    for _ in range(10):
        w.spawn_drifter("atom", 40, (0, 212, 255, 60))
    return w


def test_determinism_same_seed():
    a, b = _world(7), _world(7)
    for _ in range(120):
        a.step()
        b.step()
    for ba, bb in zip(a.bodies, b.bodies):
        assert abs(ba.x - bb.x) < 1e-9 and abs(ba.y - bb.y) < 1e-9


def test_different_seeds_diverge():
    a, b = _world(1), _world(2)
    a.step(), b.step()
    assert any(abs(ba.x - bb.x) > 1e-6 for ba, bb in zip(a.bodies, b.bodies))


def test_gravity_reaction_pulls_down():
    w = _world()
    assert w.react("Gravity!") == "heavy"
    before = [b.y for b in w.bodies]
    for _ in range(20):
        w.step()
    # under tripled gravity the mean drift must be downward
    assert sum(b.y for b in w.bodies) / len(w.bodies) > sum(before) / len(before)


def test_speed_surge_reaction():
    w = _world()
    assert w.react("speed") == "surge"
    w.step()
    assert w.speed_mult > 1.5
    for _ in range(40):
        w.step()
    assert w.speed_mult == 1.0           # effects decay


def test_unknown_word_is_inert():
    w = _world()
    assert w.react("chai") is None


def test_burst_spawns_mortal_sparks():
    w = _world()
    n0 = len(w.bodies)
    w.burst(500, 500, count=20, life=5)
    assert len(w.sparks()) == 20
    for _ in range(8):
        w.step()
    assert len(w.sparks()) == 0          # all died
    assert len(w.drifters()) == n0       # ambient field untouched


def test_spring_chain_holds_length():
    w = PhysicsWorld(1080, 1920, seed=3, gravity=(0, 0.4))
    anchor = w.spawn(500, 200)
    chain = w.add_chain(anchor, n=5, link=30, stiffness=0.8)
    for _ in range(60):
        anchor.x, anchor.px = 500, 500   # pin the anchor
        anchor.y, anchor.py = 200, 200
        w.step()
    # total chain length stays near 5 links (constraint solver works)
    total = 0.0
    prev = anchor
    for b in chain:
        total += ((b.x - prev.x) ** 2 + (b.y - prev.y) ** 2) ** 0.5
        prev = b
    assert 120 <= total <= 210
