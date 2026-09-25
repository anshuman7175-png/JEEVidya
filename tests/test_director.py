"""
Tests for Visual Director & Beat Orchestrator (§2).
"""
import pytest
from pipeline.director import AutoDirector, DirectedTimeline, StageEvent, OverlayEvent


def test_auto_director_on_astronaut_scene():
    turns = [
        {"turn_id": 1, "speaker": "girl", "text": "Chintu, space station mein astronaut float kyun karte hain?", "start_ms": 0, "end_ms": 3000},
        {"turn_id": 2, "speaker": "boy", "text": "Tumhe kya lagta hai?", "start_ms": 3100, "end_ms": 5000},
        {"turn_id": 3, "speaker": "girl", "text": "Wahan gravity khatam ho gayi!", "start_ms": 5100, "end_ms": 8000},
        {"turn_id": 4, "speaker": "boy", "text": "Wrong!", "start_ms": 8100, "end_ms": 10000},
        {"turn_id": 5, "speaker": "girl", "text": "Khatam nahi hui?", "start_ms": 10100, "end_ms": 12000},
        {"turn_id": 6, "speaker": "boy", "text": "Bilkul nahi! Earth se sirf 400 km upar hai ISS.", "start_ms": 12100, "end_ms": 15000},
        {"turn_id": 7, "speaker": "boy", "text": "Wahan gravity 8.7 m/s² hai, yaani 90%!", "start_ms": 15100, "end_ms": 18000},
        {"turn_id": 8, "speaker": "girl", "text": "Toh float kyun kar rahe hain?", "start_ms": 18100, "end_ms": 21000},
        {"turn_id": 9, "speaker": "boy", "text": "Kyunki ISS aur astronaut dono continuously GIR rahe hain!", "start_ms": 21100, "end_ms": 25000},
        {"turn_id": 10, "speaker": "girl", "text": "Matlab free fall mein weightlessness feel hota hai!", "start_ms": 25100, "end_ms": 28000},
        {"turn_id": 11, "speaker": "girl", "text": "Chintu, agar lift ki rassi toot jaye toh kya hoga?", "start_ms": 28100, "end_ms": 31000},
        {"turn_id": 12, "speaker": "boy", "text": "Woh next video mein!", "start_ms": 31100, "end_ms": 33000},
    ]

    director = AutoDirector(default_sim="orbit")
    timeline = director.direct_scene(turns, scene_meta={"topic": "Orbit & Free Fall", "belief_card_text": "GRAVITY KHATAM HO GAYI?"})

    assert isinstance(timeline, DirectedTimeline)
    assert timeline.total_duration_ms == 33000
    assert len(timeline.stage_events) >= 1
    assert len(timeline.overlay_events) >= 1

    # Check stage event mapping
    stages = [ev.stage for ev in timeline.stage_events]
    assert "DUO" in stages
    assert "LAB" in stages

    # Check overlay events
    ev_types = [ev.event_type for ev in timeline.overlay_events]
    assert "card" in ev_types       # Belief / Fix card
    assert "shatter" in ev_types    # Voronoi shatter on "Wrong!"
    assert "counter" in ev_types    # Digital counter on 8.7 / 90%
    assert "teaser" in ev_types     # Teaser on next video


def test_inline_directive_parsing():
    turns = [
        {"turn_id": 1, "speaker": "boy", "text": "> stage: LAB\n> cam: PUNCH_IN\nLook at this simulation!", "start_ms": 0, "end_ms": 2500}
    ]
    director = AutoDirector()
    timeline = director.direct_scene(turns)
    stage, cam, sim, params = timeline.get_stage_at(1000)
    assert stage == "LAB"
    assert cam == "PUNCH_IN"


def test_auto_director_cumulative_timing_when_no_start_ms():
    """Verify that turns lacking start_ms get cumulative, non-colliding timestamps."""
    turns = [
        {"turn_id": 1, "speaker": "girl", "text": "Astronaut float kyun karte hain?", "duration_ms": 2500},
        {"turn_id": 2, "speaker": "boy", "text": "Wrong! Bilkul nahi!", "duration_ms": 2000},
        {"turn_id": 3, "speaker": "boy", "text": "Agli baar dekhenge!", "duration_ms": 3000},
    ]
    director = AutoDirector()
    timeline = director.direct_scene(turns)

    # Turns should have monotonic times
    assert turns[0]["start_ms"] == 0
    assert turns[0]["end_ms"] == 2500
    assert turns[1]["start_ms"] == 2500
    assert turns[1]["end_ms"] == 4500
    assert turns[2]["start_ms"] == 4500
    assert turns[2]["end_ms"] == 7500

    # Teaser must NOT be at 300ms! It must be at the final turn!
    teaser_evs = [ev for ev in timeline.overlay_events if ev.event_type == "teaser"]
    assert len(teaser_evs) == 1
    assert teaser_evs[0].start_ms >= 4500


def test_myth_card_cleaning_and_single_trigger():
    """Verify that vocatives ('Chintu, ') are removed, and subsequent turns don't trigger 'KYA?' card."""
    turns = [
        {"turn_id": 1, "speaker": "girl", "text": "Chintu, space mein gravity nahi hoti, isliye astronaut float karte hain. Right?", "start_ms": 0, "end_ms": 3000},
        {"turn_id": 2, "speaker": "boy", "text": "Wrong!", "start_ms": 3000, "end_ms": 4500},
        {"turn_id": 3, "speaker": "girl", "text": "Kya?! Sab yahi bolte hain!", "start_ms": 4500, "end_ms": 6500},
        {"turn_id": 4, "speaker": "boy", "text": "Earth se 400 km upar gravity 8.7 m/s² hoti hai!", "start_ms": 6500, "end_ms": 9500},
        {"turn_id": 5, "speaker": "boy", "text": "Woh next video mein!", "start_ms": 9500, "end_ms": 12000}
    ]
    director = AutoDirector()
    timeline = director.direct_scene(turns)

    # There must be EXACTLY ONE belief card
    belief_cards = [ev for ev in timeline.overlay_events if ev.event_type == "card" and ev.params.get("kind") == "belief"]
    assert len(belief_cards) == 1

    card_text = belief_cards[0].params["text"]
    # Verify no vocative "CHINTU"
    assert "CHINTU" not in card_text
    # Verify not the single-word exclamation "KYA?"
    assert card_text != "KYA?"
    # Verify clean misconception question
    assert "GRAVITY NAHI HOTI" in card_text
    assert card_text.endswith("?")

    # Verify teaser card badge defaults to "[»]  UP NEXT" (no unwanted "EPISODE")
    teaser_cards = [ev for ev in timeline.overlay_events if ev.event_type == "teaser"]
    assert len(teaser_cards) == 1
    assert teaser_cards[0].params.get("badge_label") == "[»]  UP NEXT"


def test_auto_director_on_s02e02_moon_scene():
    """Verify end-to-end beat orchestration on Season 2 Episode 2."""
    import json
    import os
    from config import settings

    scene_path = os.path.join(settings.PROJECT_ROOT, "scenes", "season2", "s02e02_chaand_har_second_earth_pe_gir.json")
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    director = AutoDirector()
    timeline = director.direct_scene(scene["turns"], scene_meta=scene)

    assert isinstance(timeline, DirectedTimeline)
    assert timeline.total_duration_ms > 0

    # 1. Check stages and simulations
    stages = [ev.stage for ev in timeline.stage_events]
    assert "DUO" in stages
    assert "LAB" in stages

    sim_types = [ev.sim_type for ev in timeline.stage_events if ev.sim_type]
    assert "newton_mountain" in sim_types
    assert "orbit" in sim_types

    # 2. Check Newton's Mountain progressive modes
    nm_modes = [ev.sim_params.get("mode") for ev in timeline.stage_events if ev.sim_type == "newton_mountain"]
    assert "drop" in nm_modes
    assert "throw_short" in nm_modes
    assert "throw_far" in nm_modes
    assert "orbital_loop" in nm_modes

    # 3. Check Moon orbit mode
    orbit_modes = [ev.sim_params.get("mode") for ev in timeline.stage_events if ev.sim_type == "orbit"]
    assert "moon_orbit" in orbit_modes

    # 4. Check Overlay Events
    ev_types = [ev.event_type for ev in timeline.overlay_events]
    assert "card" in ev_types       # Belief & Fix card
    assert "counter" in ev_types    # 1.4 mm/s digital counter
    assert "notebook" in ev_types   # a = v^2/R notebook card
    assert "teaser" in ev_types     # Escape velocity teaser

    # 5. Check counter parameters
    counter_ev = next(ev for ev in timeline.overlay_events if ev.event_type == "counter")
    assert counter_ev.params["target_val"] == 1.4
    assert counter_ev.params["unit"] == "mm/s"

    # 6. Check teaser parameters
    teaser_ev = next(ev for ev in timeline.overlay_events if ev.event_type == "teaser")
    assert teaser_ev.params["next_episode_num"] == 3
    assert "ESCAPE VELOCITY" in teaser_ev.params["question"]


def test_auto_director_on_s02e03_escape_velocity_scene():
    """Verify end-to-end beat orchestration on Season 2 Episode 3 (Escape Velocity)."""
    import json
    import os
    from config import settings

    scene_path = os.path.join(settings.PROJECT_ROOT, "scenes", "season2", "s02e03_escape_velocity_11_2_km_s.json")
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)

    director = AutoDirector()
    timeline = director.direct_scene(scene["turns"], scene_meta=scene)

    assert isinstance(timeline, DirectedTimeline)
    assert timeline.total_duration_ms > 0

    # 1. Check stages and simulations
    stages = [ev.stage for ev in timeline.stage_events]
    assert "DUO" in stages
    assert "LAB" in stages

    sim_types = [ev.sim_type for ev in timeline.stage_events if ev.sim_type]
    assert "escape_velocity" in sim_types

    # 2. Check Escape Velocity progressive modes
    ev_modes = [ev.sim_params.get("mode") for ev in timeline.stage_events if ev.sim_type == "escape_velocity"]
    assert "earth_escape" in ev_modes
    assert "gravity_well_comparison" in ev_modes
    assert "atmospheric_escape" in ev_modes

    # 3. Check Overlay Events
    ev_types = [ev.event_type for ev in timeline.overlay_events]
    assert "card" in ev_types       # Belief & Fix card
    assert "counter" in ev_types    # 11.2 km/s and/or 2.4 km/s counters
    assert "notebook" in ev_types   # Formula card
    assert "teaser" in ev_types     # Next episode teaser

    # 4. Check counter parameters
    counters = [ev for ev in timeline.overlay_events if ev.event_type == "counter"]
    assert len(counters) >= 1
    target_vals = [c.params["target_val"] for c in counters]
    assert 11.2 in target_vals

    # 5. Check teaser parameters
    teaser_ev = next(ev for ev in timeline.overlay_events if ev.event_type == "teaser")
    assert teaser_ev.params["next_episode_num"] == 4
    assert "SHOCK" in teaser_ev.params["question"]



