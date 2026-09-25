"""
JEEVidya — Visual Director & Beat Orchestration Engine (Visual Bible v1 §2 & §5)
══════════════════════════════════════════════════════════════════════════════════
Orchestrates:
  1. Stage transitions: DUO (dialogue), LAB (sim + busts), PROOF (fullscreen hero).
  2. Camera framing: NORMAL, WIDE, PUNCH_IN.
  3. Signature Overlays:
      - Belief Card & Fix Card
      - 18-shard Voronoi Glass Shatter on "Wrong!"
      - Live Digital Counter
      - Gudiya's Notebook with LaTeX export
      - Teaser Card with seamless infinite loop
  4. Procedural Simulations: Orbit, Elevator, etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StageEvent:
    stage: str  # "DUO", "LAB", "PROOF"
    start_ms: int
    end_ms: int
    camera: str = "NORMAL"
    sim_type: Optional[str] = None
    sim_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverlayEvent:
    event_type: str  # "card", "shatter", "counter", "notebook", "teaser"
    start_ms: int
    end_ms: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectedTimeline:
    total_duration_ms: int
    stage_events: List[StageEvent] = field(default_factory=list)
    overlay_events: List[OverlayEvent] = field(default_factory=list)

    def get_stage_at(self, t_ms: int) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
        """Return (stage_name, camera_mode, sim_type, sim_params) at given millisecond."""
        for ev in self.stage_events:
            if ev.start_ms <= t_ms < ev.end_ms:
                return ev.stage, ev.camera, ev.sim_type, ev.sim_params
        return "DUO", "NORMAL", None, {}

    def get_active_overlays(self, t_ms: int) -> List[OverlayEvent]:
        """Return all active overlays at given millisecond."""
        return [ev for ev in self.overlay_events if ev.start_ms <= t_ms < ev.end_ms]


class AutoDirector:
    """
    Analyzes script dialogue turns, detected beats, and emotion tags to generate
    a comprehensive Visual Bible timeline.
    """

    def __init__(self, default_sim: Optional[str] = None):
        self.default_sim = default_sim

    def direct_scene(self, turns: List[Dict[str, Any]],
                     scene_meta: Optional[Dict[str, Any]] = None) -> DirectedTimeline:
        """
        Produce a DirectedTimeline from a list of dialogue turns.
        Each turn is expected to have 'start_ms', 'end_ms', 'speaker', and 'text'.
        """
        if not turns:
            return DirectedTimeline(total_duration_ms=0)

        # Compute cumulative timeline boundaries for all turns
        cursor_ms = 0
        computed_spans: List[Tuple[int, int]] = []
        for t in turns:
            if "start_ms" in t and t["start_ms"] is not None and int(t.get("start_ms", 0)) >= cursor_ms:
                s_ms = int(t["start_ms"])
                e_ms = int(t.get("end_ms", s_ms + int(t.get("duration_ms", 2200))))
            else:
                s_ms = cursor_ms
                dur = int(t.get("duration_ms", 2200))
                e_ms = cursor_ms + dur
            cursor_ms = max(cursor_ms, e_ms)
            computed_spans.append((s_ms, e_ms))

        total_duration_ms = cursor_ms
        stage_events: List[StageEvent] = []
        overlay_events: List[OverlayEvent] = []

        scene_meta = scene_meta or {}
        sim_name = scene_meta.get("simulation") or self.default_sim

        # Detect topic if not explicitly given
        topic = scene_meta.get("topic", "")
        if not sim_name:
            # Infer from topic or text
            full_text = " ".join(t.get("text", "") for t in turns).lower()
            if any(k in full_text for k in ["escape", "escape velocity", "11.2", "chhodne"]):
                sim_name = "escape_velocity"
            elif any(k in full_text for k in ["mountain", "ball", "phenko", "seedha chhodo", "zor se"]):
                sim_name = "newton_mountain"
            elif "orbit" in full_text or "iss" in full_text or "astronaut" in full_text or "float" in full_text:
                sim_name = "orbit"
            elif "lift" in full_text or "elevator" in full_text or "weighing" in full_text:
                sim_name = "elevator"

        num_turns = len(turns)
        shatter_triggered = False
        belief_card_triggered = False
        golden_rule_triggered = False
        counter_triggered = False
        counter_values_seen = set()
        last_lab_sim: Optional[str] = None

        for i, turn in enumerate(turns):
            start_ms, end_ms = computed_spans[i]
            turn["start_ms"] = start_ms
            turn["end_ms"] = end_ms
            text = turn.get("text", "")
            speaker = turn.get("speaker", "boy").lower()
            dur_ms = end_ms - start_ms

            # ── 1. Check for Inline Directives First ──
            has_directives = ">" in text
            clean_text, directives = self._parse_directives(text)

            stage_name = directives.get("stage")
            cam_mode = directives.get("cam", "NORMAL")

            # ── 2. Automatic Beat Mapping (Visual Bible 7 Beats) ──
            if not stage_name:
                progress = i / max(1, num_turns - 1)
                clean_lower = clean_text.lower()

                # Beat 1 & 2: Hook & Misconception (First 20% of turns)
                if (i <= 1 or progress < 0.20) and not belief_card_triggered:
                    stage_name = "DUO"
                    # If this turn states the misconception, show Belief Card
                    if ("kyun" in clean_lower or "udte" in clean_lower or
                            "gravity" in clean_lower or "?" in clean_text or
                            "nahi hoti" in clean_lower or "float" in clean_lower or
                            "miss" in clean_lower):
                        card_text = scene_meta.get("belief_card_text")
                        if not card_text:
                            # Strip leading vocatives ("Chintu, ", "Gudiya, ", "Bhaiya, ", etc.)
                            c_text = re.sub(r'^(?:chintu|gudiya|didi|bhaiya|suno|bhai|arrey|dekho)\s*,\s*', '', clean_text, flags=re.I).strip()
                            # Strip trailing tag questions (", right?", ", hai na?", ", na?")
                            c_text = re.sub(r'[,.]?\s*(?:right|hai na|kyun|batao|na)\s*\??$', '?', c_text, flags=re.I).strip()
                            # If there's an 'isliye' clause, isolate the misconception premise
                            premise = re.split(r'[,.]?\s*isliye', c_text, flags=re.I)[0].strip()
                            if len(premise) >= 15:
                                c_text = premise.rstrip('?.') + "?"
                            elif "?" in c_text:
                                c_text = c_text.split("?")[0].strip() + "?"
                            card_text = c_text
                            if len(card_text) > 42:
                                card_text = card_text[:40].rsplit(' ', 1)[0] + "...?"

                        # Strip emojis from card text to avoid tofu glyph boxes
                        clean_card = re.sub(r'[\U00010000-\U0010ffff]', '', card_text).strip()
                        # Minimum length guard: reject 1-word exclamations like "Kya?"
                        if len(clean_card) >= 12:
                            belief_card_triggered = True
                            overlay_events.append(OverlayEvent(
                                event_type="card",
                                start_ms=start_ms + 200,
                                end_ms=min(end_ms + 400, start_ms + 3200),
                                params={"text": clean_card.upper(), "kind": "belief"}
                            ))

                # Beat 3: The Shatter ("Wrong!", "Nahi!")
                elif (("wrong" in clean_lower or "bilkul nahi" in clean_lower or
                       clean_text.strip().lower() == "zero") and not shatter_triggered):
                    stage_name = "DUO"
                    cam_mode = "PUNCH_IN"
                    shatter_triggered = True
                    overlay_events.append(OverlayEvent(
                        event_type="shatter",
                        start_ms=start_ms + 100,
                        end_ms=start_ms + 1800,
                        params={"text": "WRONG!"}
                    ))

                # Beat 7: The Loop / Teaser (STRICTLY Final turn or when explicitly saying "next video" / "agli baar")
                elif i == num_turns - 1 or "next video" in clean_lower or "agli baar" in clean_lower:
                    stage_name = "DUO"
                    teaser_q = scene_meta.get("teaser_question", "AGLI BAAR: Lift ki rassi tooti toh...?")
                    clean_teaser = re.sub(r'[\U00010000-\U0010ffff]', '', teaser_q).strip()
                    next_ep = scene_meta.get("next_episode_num") or scene_meta.get("episode_number")
                    badge = f"[»]  NEXT: EPISODE #{next_ep}" if next_ep else "[»]  UP NEXT"
                    pinned = scene_meta.get("pinned_comment") or "Answer in the comments! 👇"
                    clean_pinned = re.sub(r'[\U00010000-\U0010ffff]', '', pinned).strip()
                    # Ensure no prior overlay (e.g. notebook) overlaps with the teaser card
                    for ev in overlay_events:
                        if ev.end_ms > start_ms:
                            ev.end_ms = max(ev.start_ms + 400, start_ms - 50)
                    overlay_events.append(OverlayEvent(
                        event_type="teaser",
                        start_ms=start_ms + 150,
                        end_ms=end_ms,
                        params={
                            "question": clean_teaser,
                            "loop_cue": clean_pinned,
                            "badge_label": badge,
                            "next_episode_num": next_ep
                        }
                    ))

                # Post-Proof Resolution: Once Golden Rule is revealed, maintain DUO for synthesis,
                # UNLESS the character explicitly asks about the upcoming teaser topic (e.g. escape velocity setup)
                elif golden_rule_triggered:
                    if any(k in clean_lower for k in ["chhodne", "escape", "kitni speed", "agli video", "next episode"]):
                        stage_name = "LAB"
                    else:
                        stage_name = "DUO"

                # Beat 4 & 5: Mechanism & Proof (Middle turns or physical explanations)
                elif (0.20 <= progress < 0.88 or
                      any(k in clean_lower for k in [
                          "space station", "400", "percent", "gir rahe", "lift", "rassi", "free fall",
                          "miss", "bhaag", "curve", "zameen", "ball", "phenko", "seedha", "zor se", "girti",
                          "chaand", "moon", "1.4", "millimetre", "kilometre",
                          "escape", "velocity", "11.2", "2.4", "dilli", "mumbai", "molecules", "gas", "hawa", "atmosphere", "venus", "chhodne"
                      ])):
                    # Check if this turn is an explicit Golden Rule conclusion card
                    if ("rule" in clean_lower or "formula" in clean_lower or
                        ("matlab" in clean_lower and ("weightlessness" in clean_lower or "miss" in clean_lower or "simple" in clean_lower or "escape" in clean_lower or "hawa" in clean_lower) and "lift" not in clean_lower)):
                        stage_name = "DUO"
                        golden_rule_triggered = True
                        rule_text = scene_meta.get("golden_rule", "FREE FALL MEIN WEIGHTLESSNESS HOTA HAI")
                        fix_end = start_ms + 2400
                        overlay_events.append(OverlayEvent(
                            event_type="card",
                            start_ms=start_ms + 100,
                            end_ms=fix_end,
                            params={"text": rule_text.upper(), "kind": "fix"}
                        ))
                        # Notebook overlay follows after Fix Card
                        formula = scene_meta.get("formula", "g = \\frac{GM}{(R+h)^2}")
                        takeaway = scene_meta.get("takeaway", "Gravity is still 89% active at ISS!")
                        overlay_events.append(OverlayEvent(
                            event_type="notebook",
                            start_ms=fix_end + 150,
                            end_ms=fix_end + 3200,
                            params={"formula": formula, "takeaway": takeaway, "topic": topic or "Mechanics"}
                        ))
                    else:
                        stage_name = "LAB"
                        # If mentioning numbers, values, or formula, trigger Counter
                        if any(c in clean_lower for c in [
                            "8.7", "9.8", "90%", "90 percent", "400", "0.0", "zero", "100",
                            "1.4", "1 point 4", "millimetre", "11.2", "11 point 2", "2.4", "2 point 4"
                        ]):
                            if any(c in clean_lower for c in ["11.2", "11 point 2"]) and "11.2" not in counter_values_seen:
                                counter_values_seen.add("11.2")
                                counter_triggered = True
                                overlay_events.append(OverlayEvent(
                                    event_type="counter",
                                    start_ms=start_ms + 150,
                                    end_ms=min(end_ms, start_ms + 2800),
                                    params={
                                        "start_val": 0.0, "target_val": 11.2,
                                        "unit": "km/s", "label": "EARTH ESCAPE VELOCITY",
                                        "subtitle": "40,320 km/h - Delhi to Mumbai in 2 min",
                                        "format_str": "{val:.1f} {unit}"
                                    }
                                ))
                            elif any(c in clean_lower for c in ["2.4", "2 point 4"]) and "2.4" not in counter_values_seen:
                                counter_values_seen.add("2.4")
                                counter_triggered = True
                                overlay_events.append(OverlayEvent(
                                    event_type="counter",
                                    start_ms=start_ms + 150,
                                    end_ms=min(end_ms, start_ms + 2800),
                                    params={
                                        "start_val": 11.2, "target_val": 2.4,
                                        "unit": "km/s", "label": "MOON'S ESCAPE VELOCITY",
                                        "subtitle": "4.7x lower than Earth",
                                        "format_str": "{val:.1f} {unit}"
                                    }
                                ))
                            elif not counter_triggered and any(c in clean_lower for c in ["1.4", "1 point 4", "millimetre", "moon", "chaand"]):
                                counter_triggered = True
                                overlay_events.append(OverlayEvent(
                                    event_type="counter",
                                    start_ms=start_ms + 150,
                                    end_ms=min(end_ms, start_ms + 2800),
                                    params={
                                        "start_val": 0.0, "target_val": 1.4,
                                        "unit": "mm/s", "label": "MOON'S FREE-FALL SPEED",
                                        "subtitle": "Earth's gravity at 3,84,400 km",
                                        "format_str": "{val:.1f} {unit}"
                                    }
                                ))
                            elif not counter_triggered:
                                counter_triggered = True
                                overlay_events.append(OverlayEvent(
                                    event_type="counter",
                                    start_ms=start_ms + 150,
                                    end_ms=min(end_ms, start_ms + 2800),
                                    params={
                                        "start_val": 9.8, "target_val": 8.7,
                                        "unit": "m/s²", "label": "GRAVITY AT ISS (400 KM)",
                                        "subtitle": "89% of Earth surface gravity",
                                        "format_str": "{val:.1f} {unit}"
                                    }
                                ))
                else:
                    stage_name = "DUO"

            # ── 3. Assign Dynamic Simulation for LAB Stage ──
            turn_sim: Optional[str] = None
            turn_params: Dict[str, Any] = {}
            if stage_name == "LAB":
                clean_lower = clean_text.lower()
                if any(w in clean_lower for w in ["lift", "elevator", "rassi", "taar", "snapped"]):
                    turn_sim = "elevator"
                    # Compute snap_time from word-level events: the cable
                    # should visually snap when "toot"/"snap"/"break" is spoken
                    snap_time = 1.5  # sensible default
                    word_events = turn.get("word_events", [])
                    for we in word_events:
                        w_lower = we.get("text", "").lower().strip("?!.,;:")
                        if w_lower in ("toot", "tut", "snap", "break", "snapped", "tooti"):
                            snap_time = we["start_ms"] / 1000.0
                            break
                    turn_params = {"mode": "cable_snap", "snap_time": snap_time, "stage_start_ms": start_ms}
                elif "free fall" in clean_lower and not any(w in clean_lower for w in ["moon", "chaand", "ball", "phenko", "orbit", "escape"]):
                    turn_sim = "elevator"
                    turn_params = {"mode": "free_fall", "stage_start_ms": start_ms}
                elif (sim_name == "escape_velocity" or
                      any(w in clean_lower for w in ["escape velocity", "chhodne", "dilli se mumbai", "11.2", "2.4", "molecules", "hawa kyun nahi", "atmosphere", "venus"]) or
                      (last_lab_sim == "escape_velocity" and not any(w in clean_lower for w in ["space station", "iss", "lift", "ball", "mountain"]))):
                    turn_sim = "escape_velocity"
                    if any(w in clean_lower for w in ["molecules", "gas", "hawa", "garmi", "atmosphere", "venus", "space mein bhaag"]):
                        ev_mode = "atmospheric_escape"
                    elif any(w in clean_lower for w in ["moon pe", "chhota hai", "gravity kam", "2.4", "2 point 4"]):
                        ev_mode = "gravity_well_comparison"
                    else:
                        ev_mode = "earth_escape"
                    turn_params = {"mode": ev_mode, "stage_start_ms": start_ms}
                elif any(w in clean_lower for w in ["moon", "chaand", "1.4", "millimetre"]) or (last_lab_sim == "orbit" and any(w in clean_lower for w in ["kilometre", "arab saal", "bhaag"])):
                    turn_sim = "orbit"
                    turn_params = {"mode": "moon_orbit", "target": "moon", "stage_start_ms": start_ms}
                elif (sim_name == "newton_mountain" or
                      any(w in clean_lower for w in ["ball", "phenko", "seedha", "zor se phenko", "pahunchegi hi nahi"]) or
                      (last_lab_sim == "newton_mountain" and not any(w in clean_lower for w in ["space station", "iss", "lift", "moon", "chaand"]))):
                    turn_sim = "newton_mountain"
                    # Determine progressive trajectory mode based on narrative beat
                    if "seedha" in clean_lower or ("ball" in clean_lower and "neeche" in clean_lower):
                        nm_mode = "drop"
                    elif any(w in clean_lower for w in ["aage jaake girti", "aage phenko"]) or ("zor se" in clean_lower and "itni" not in clean_lower and "aur aage" not in clean_lower):
                        nm_mode = "throw_short"
                    elif any(w in clean_lower for w in ["aur aage", "itni zor", "curve ho"]):
                        nm_mode = "throw_far"
                    elif any(w in clean_lower for w in ["pahunchegi hi nahi", "chakkar", "orbit", "girti rahegi"]):
                        nm_mode = "orbital_loop"
                    else:
                        nm_mode = "throw_short"
                    turn_params = {"mode": nm_mode, "stage_start_ms": start_ms}
                elif any(w in clean_lower for w in ["miss", "zameen", "side", "bhaag", "curve", "curve ho", "gol"]):
                    turn_sim = "orbit"
                    turn_params = {"mode": "newton_curvature", "stage_start_ms": start_ms}
                else:
                    turn_sim = sim_name or "orbit"
                    if turn_sim == "orbit":
                        turn_params = {"mode": "circular", "show_vectors": True, "gravity_vector": True, "stage_start_ms": start_ms}
                    elif turn_sim == "newton_mountain":
                        turn_params = {"mode": "throw_short", "stage_start_ms": start_ms}
                    elif turn_sim == "escape_velocity":
                        turn_params = {"mode": "earth_escape", "stage_start_ms": start_ms}
                    else:
                        turn_params = {"mode": "static", "stage_start_ms": start_ms}
                last_lab_sim = turn_sim
            else:
                turn_sim = None
                turn_params = {}



            # Add stage event
            stage_events.append(StageEvent(
                stage=stage_name,
                start_ms=start_ms,
                end_ms=end_ms,
                camera=cam_mode,
                sim_type=turn_sim,
                sim_params=turn_params
            ))

        # Coalesce continuous stage events if needed
        coalesced_stages = self._coalesce_stages(stage_events, total_duration_ms)

        return DirectedTimeline(
            total_duration_ms=total_duration_ms,
            stage_events=coalesced_stages,
            overlay_events=overlay_events
        )

    def _parse_directives(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Extract inline directive comments like '> stage: LAB' or '> cam: PUNCH_IN'."""
        directives: Dict[str, str] = {}
        lines = text.split("\n")
        clean_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(">"):
                match = re.match(r"^>\s*([a-zA-Z0-9_\+]+)\s*[:=]\s*(.*)$", stripped)
                if match:
                    key = match.group(1).lower()
                    val = match.group(2).strip()
                    directives[key] = val
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines), directives

    def _coalesce_stages(self, events: List[StageEvent], total_ms: int) -> List[StageEvent]:
        """Merge adjacent events with identical stage, camera, and simulation mode."""
        if not events:
            return [StageEvent(stage="DUO", start_ms=0, end_ms=total_ms)]

        result: List[StageEvent] = []
        cur = events[0]

        for nxt in events[1:]:
            same_sim = (cur.sim_type == nxt.sim_type and
                        (cur.sim_params.get("mode") if cur.sim_params else None) ==
                        (nxt.sim_params.get("mode") if nxt.sim_params else None))
            if cur.stage == nxt.stage and cur.camera == nxt.camera and same_sim:
                cur.end_ms = nxt.end_ms
            else:
                result.append(cur)
                cur = nxt
        result.append(cur)
        return result
