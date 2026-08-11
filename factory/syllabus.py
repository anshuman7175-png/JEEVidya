"""
JEEVidya — Syllabus Coverage Graph (Terminal Plan, Part XIX)
═══════════════════════════════════════════════════════════
Topic selection, made mechanical.

A million-sub education channel cannot pick topics by vibe. This module
holds the JEE syllabus as a DEPENDENCY DAG (topic → prerequisites), a
coverage ledger (what has been taught, when, how often), and a
spaced-repetition scheduler for revisits. Combined with audience demand
from `agents/comment_miner.py`, `plan_next()` emits a ranked, ready-to-
teach queue where every entry satisfies:

    1. its prerequisites are already covered (no orphan lessons),
    2. it is either NEW, or DUE for a spaced revisit (1/3/7/21/60 days),
    3. it is ranked by measured audience demand, not guesswork.

The DAG is asserted acyclic at import-time cost O(V+E) by `validate()`
(a cycle would make "prerequisites covered" unsatisfiable forever — Law
1: make that unrepresentable).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from config import settings

DB_PATH = os.path.join(settings.PROJECT_ROOT, ".cache", "flywheel.db")

# Spaced-repetition ladder in days: a topic re-taught from a new angle at
# these intervals is how a channel builds retention without new syllabus.
REVISIT_LADDER_DAYS = (1.0, 3.0, 7.0, 21.0, 60.0, 150.0)

_WORD = re.compile(r"[a-z\u0900-\u097f]+")


# ═══════════════════════════════════════════
# The graph (subject → chapter → topic, with prerequisites)
# ═══════════════════════════════════════════
# keywords are Hinglish-aware on purpose: comments arrive in three scripts.

TOPICS: Dict[str, Dict[str, Any]] = {
    # ── Physics: Mechanics ──
    "units_dimensions": dict(
        subject="physics", chapter="Units & Dimensions",
        title="Dimensional analysis", prereqs=[],
        keywords=["dimension", "unit", "si", "dimensional", "matra"]),
    "kinematics_1d": dict(
        subject="physics", chapter="Kinematics",
        title="Motion in a straight line", prereqs=["units_dimensions"],
        keywords=["velocity", "acceleration", "displacement", "kinematics",
                  "speed", "vega", "chaal", "graph"]),
    "projectile": dict(
        subject="physics", chapter="Kinematics",
        title="Projectile motion", prereqs=["kinematics_1d"],
        keywords=["projectile", "range", "trajectory", "prakshep", "angle"]),
    "newton_laws": dict(
        subject="physics", chapter="Laws of Motion",
        title="Newton's laws & free-body diagrams", prereqs=["kinematics_1d"],
        keywords=["newton", "force", "fbd", "tension", "normal", "bal",
                  "friction", "gharshan"]),
    "circular_motion": dict(
        subject="physics", chapter="Laws of Motion",
        title="Circular motion & banking", prereqs=["newton_laws"],
        keywords=["circular", "centripetal", "banking", "vartul", "curve"]),
    "work_energy": dict(
        subject="physics", chapter="Work, Power, Energy",
        title="Work–energy theorem", prereqs=["newton_laws"],
        keywords=["work", "energy", "power", "kinetic", "potential", "urja"]),
    "momentum": dict(
        subject="physics", chapter="Centre of Mass",
        title="Momentum & collisions", prereqs=["newton_laws"],
        keywords=["momentum", "collision", "impulse", "sanvég", "elastic"]),
    "rotation": dict(
        subject="physics", chapter="Rotational Motion",
        title="Torque & moment of inertia", prereqs=["momentum", "work_energy"],
        keywords=["torque", "inertia", "angular", "rolling", "ghurnan"]),
    "gravitation": dict(
        subject="physics", chapter="Gravitation",
        title="Gravitation & escape velocity",
        prereqs=["circular_motion", "work_energy"],
        keywords=["gravity", "gravitation", "escape velocity", "orbit",
                  "satellite", "gurutvakarshan", "kepler"]),
    "shm": dict(
        subject="physics", chapter="Oscillations",
        title="Simple harmonic motion", prereqs=["work_energy"],
        keywords=["shm", "oscillation", "pendulum", "spring", "dolan"]),
    "waves": dict(
        subject="physics", chapter="Waves",
        title="Waves, beats & Doppler", prereqs=["shm"],
        keywords=["wave", "beat", "doppler", "sound", "tarang", "frequency"]),
    # ── Physics: Thermo, EM, Modern ──
    "thermodynamics": dict(
        subject="physics", chapter="Thermodynamics",
        title="First law & processes", prereqs=["work_energy"],
        keywords=["thermodynamics", "isothermal", "adiabatic", "entropy",
                  "heat", "ushma", "carnot"]),
    "electrostatics": dict(
        subject="physics", chapter="Electrostatics",
        title="Coulomb, field & potential", prereqs=["work_energy"],
        keywords=["charge", "coulomb", "electric field", "potential",
                  "gauss", "vidyut", "capacitor"]),
    "current_electricity": dict(
        subject="physics", chapter="Current Electricity",
        title="Circuits & Kirchhoff", prereqs=["electrostatics"],
        keywords=["current", "resistance", "kirchhoff", "circuit", "dhara",
                  "ohm", "wheatstone"]),
    "magnetism": dict(
        subject="physics", chapter="Magnetic Effects",
        title="Magnetic force & fields", prereqs=["current_electricity"],
        keywords=["magnetic", "biot", "solenoid", "lorentz", "chumbak"]),
    "emi_ac": dict(
        subject="physics", chapter="EMI & AC",
        title="Induction & AC circuits", prereqs=["magnetism"],
        keywords=["induction", "faraday", "lenz", "ac", "impedance",
                  "resonance", "transformer"]),
    "optics": dict(
        subject="physics", chapter="Optics",
        title="Ray & wave optics", prereqs=["waves"],
        keywords=["lens", "mirror", "refraction", "interference",
                  "diffraction", "prakash", "young"]),
    "modern_physics": dict(
        subject="physics", chapter="Modern Physics",
        title="Photoelectric, atoms & nuclei",
        prereqs=["electrostatics", "optics"],
        keywords=["photoelectric", "bohr", "nucleus", "de broglie",
                  "radioactive", "parmanu", "photon"]),
    # ── Chemistry ──
    "mole_concept": dict(
        subject="chemistry", chapter="Some Basic Concepts",
        title="Mole concept & stoichiometry", prereqs=[],
        keywords=["mole", "stoichiometry", "molarity", "equivalent",
                  "concentration", "mol"]),
    "atomic_structure": dict(
        subject="chemistry", chapter="Atomic Structure",
        title="Quantum numbers & orbitals", prereqs=["mole_concept"],
        keywords=["orbital", "quantum number", "aufbau", "hund",
                  "electronic configuration", "parmanu sanrachna"]),
    "periodicity": dict(
        subject="chemistry", chapter="Periodic Table",
        title="Periodic trends", prereqs=["atomic_structure"],
        keywords=["periodic", "ionization", "electronegativity",
                  "atomic radius", "trend"]),
    "bonding": dict(
        subject="chemistry", chapter="Chemical Bonding",
        title="Hybridisation, VSEPR & MOT", prereqs=["periodicity"],
        keywords=["bond", "hybridisation", "vsepr", "mot", "dipole",
                  "resonance", "bandh"]),
    "equilibrium": dict(
        subject="chemistry", chapter="Equilibrium",
        title="Chemical & ionic equilibrium", prereqs=["mole_concept"],
        keywords=["equilibrium", "kc", "kp", "ph", "buffer", "solubility",
                  "samyavastha", "le chatelier"]),
    "thermochem": dict(
        subject="chemistry", chapter="Thermodynamics",
        title="Enthalpy, entropy & Gibbs", prereqs=["mole_concept"],
        keywords=["enthalpy", "entropy", "gibbs", "hess", "bond energy"]),
    "electrochem": dict(
        subject="chemistry", chapter="Electrochemistry",
        title="Cells & Nernst", prereqs=["equilibrium", "thermochem"],
        keywords=["electrochemistry", "nernst", "cell", "electrolysis",
                  "emf", "galvanic"]),
    "chemical_kinetics": dict(
        subject="chemistry", chapter="Kinetics",
        title="Rate laws & order", prereqs=["equilibrium"],
        keywords=["kinetics", "rate", "order", "half life", "arrhenius"]),
    "goc": dict(
        subject="chemistry", chapter="Organic Chemistry",
        title="GOC: inductive, resonance, acidity", prereqs=["bonding"],
        keywords=["goc", "resonance", "inductive", "acidity", "basicity",
                  "carbocation", "stability", "organic"]),
    "reaction_mechanisms": dict(
        subject="chemistry", chapter="Organic Chemistry",
        title="SN1/SN2, E1/E2 mechanisms", prereqs=["goc"],
        keywords=["sn1", "sn2", "e1", "e2", "mechanism", "substitution",
                  "elimination", "nucleophile"]),
    "coordination": dict(
        subject="chemistry", chapter="Coordination Compounds",
        title="Coordination compounds & CFT", prereqs=["bonding"],
        keywords=["coordination", "ligand", "cft", "isomerism",
                  "complex", "crystal field"]),
    # ── Mathematics ──
    "quadratics": dict(
        subject="maths", chapter="Algebra",
        title="Quadratic equations", prereqs=[],
        keywords=["quadratic", "root", "discriminant", "samikaran"]),
    "sequences": dict(
        subject="maths", chapter="Algebra",
        title="Sequences & series", prereqs=["quadratics"],
        keywords=["ap", "gp", "series", "sequence", "shreni", "sum"]),
    "binomial": dict(
        subject="maths", chapter="Algebra",
        title="Binomial theorem", prereqs=["sequences"],
        keywords=["binomial", "expansion", "coefficient"]),
    "permutations": dict(
        subject="maths", chapter="Combinatorics",
        title="Permutations & combinations", prereqs=["quadratics"],
        keywords=["permutation", "combination", "ncr", "arrangement"]),
    "probability": dict(
        subject="maths", chapter="Probability",
        title="Probability & Bayes", prereqs=["permutations"],
        keywords=["probability", "bayes", "conditional", "sambhavna"]),
    "trigonometry": dict(
        subject="maths", chapter="Trigonometry",
        title="Trigonometric identities & equations", prereqs=["quadratics"],
        keywords=["trigonometry", "sin", "cos", "identity", "tan",
                  "trikonmiti"]),
    "functions": dict(
        subject="maths", chapter="Calculus",
        title="Functions, limits & continuity",
        prereqs=["trigonometry", "quadratics"],
        keywords=["function", "limit", "continuity", "domain", "range",
                  "seema", "fankshan"]),
    "differentiation": dict(
        subject="maths", chapter="Calculus",
        title="Differentiation & applications", prereqs=["functions"],
        keywords=["derivative", "differentiation", "tangent", "maxima",
                  "minima", "avkalan"]),
    "integration": dict(
        subject="maths", chapter="Calculus",
        title="Integration techniques", prereqs=["differentiation"],
        keywords=["integration", "integral", "definite", "substitution",
                  "samakalan", "area"]),
    "differential_eq": dict(
        subject="maths", chapter="Calculus",
        title="Differential equations", prereqs=["integration"],
        keywords=["differential equation", "variable separable",
                  "linear differential", "order degree"]),
    "coordinate_geometry": dict(
        subject="maths", chapter="Coordinate Geometry",
        title="Straight lines & circles", prereqs=["quadratics"],
        keywords=["line", "circle", "slope", "locus", "nirdeshank"]),
    "conics": dict(
        subject="maths", chapter="Coordinate Geometry",
        title="Parabola, ellipse & hyperbola",
        prereqs=["coordinate_geometry"],
        keywords=["parabola", "ellipse", "hyperbola", "conic", "eccentricity"]),
    "vectors_3d": dict(
        subject="maths", chapter="Vectors & 3D",
        title="Vectors & three-dimensional geometry",
        prereqs=["trigonometry", "coordinate_geometry"],
        keywords=["vector", "dot product", "cross product", "plane",
                  "3d", "sadish"]),
    "matrices": dict(
        subject="maths", chapter="Algebra",
        title="Matrices & determinants", prereqs=["quadratics"],
        keywords=["matrix", "determinant", "adjoint", "inverse", "rank"]),
    "complex_numbers": dict(
        subject="maths", chapter="Algebra",
        title="Complex numbers", prereqs=["quadratics", "trigonometry"],
        keywords=["complex", "argand", "modulus", "argument", "iota"]),
}


def validate(graph: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Assert the DAG is acyclic and every prerequisite exists.

    A cycle would make `ready()` permanently false for its members — an
    unteachable topic that no gate would otherwise notice.
    """
    graph = graph or TOPICS
    for tid, node in graph.items():
        for p in node.get("prereqs", []):
            if p not in graph:
                raise ValueError(f"syllabus: {tid} requires unknown '{p}'")
    state: Dict[str, int] = {}

    def visit(node: str, stack: Tuple[str, ...]) -> None:
        st = state.get(node, 0)
        if st == 1:
            raise ValueError("syllabus cycle: "
                             + " → ".join(stack + (node,)))
        if st == 2:
            return
        state[node] = 1
        for p in graph[node].get("prereqs", []):
            visit(p, stack + (node,))
        state[node] = 2

    for tid in graph:
        visit(tid, ())


def topological_order(graph: Optional[Dict[str, Dict[str, Any]]] = None
                      ) -> List[str]:
    """Prerequisites first — the order a from-scratch channel should teach."""
    graph = graph or TOPICS
    validate(graph)
    seen: Set[str] = set()
    order: List[str] = []

    def visit(node: str) -> None:
        if node in seen:
            return
        for p in graph[node].get("prereqs", []):
            visit(p)
        seen.add(node)
        order.append(node)

    for tid in sorted(graph):
        visit(tid)
    return order


def _tokens(text: str) -> Set[str]:
    return set(_WORD.findall(str(text or "").lower()))


# ═══════════════════════════════════════════
# Coverage ledger + scheduler
# ═══════════════════════════════════════════

class Syllabus:
    """Coverage ledger over the topic DAG, with spaced revisits."""

    def __init__(self, db_path: str = DB_PATH,
                 graph: Optional[Dict[str, Dict[str, Any]]] = None):
        self.graph = graph or TOPICS
        validate(self.graph)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS coverage (
                topic TEXT PRIMARY KEY,
                times INTEGER NOT NULL DEFAULT 0,
                last_taught REAL,
                last_video TEXT
            );
        """)
        self.db.commit()

    # ─── matching (comment text → topic) ───────────────────

    def match(self, text: str) -> Dict[str, Any]:
        """Best topic for a blob of audience text, with confidence in
        [0,1]. Multi-word keywords are matched as substrings; single
        words as tokens, so "escape velocity" beats a stray "velocity"."""
        low = str(text or "").lower()
        toks = _tokens(low)
        best_id, best_score = None, 0.0
        for tid, node in self.graph.items():
            score = 0.0
            for kw in node.get("keywords", []):
                if " " in kw:
                    if kw in low:
                        score += 2.0
                elif kw in toks:
                    score += 1.0
            if str(node.get("title", "")).lower() in low:
                score += 2.0
            if score > best_score:
                best_id, best_score = tid, score
        if best_id is None:
            return {"topic": None, "confidence": 0.0}
        node = self.graph[best_id]
        return {
            "topic": best_id,
            "title": node["title"],
            "chapter": node["chapter"],
            "subject": node["subject"],
            "prerequisites": list(node.get("prereqs", [])),
            "confidence": round(min(1.0, best_score / 4.0), 3),
        }

    # ─── coverage ──────────────────────────────────────────

    def mark_taught(self, topic: str, video_id: str = "",
                    when: Optional[float] = None) -> None:
        if topic not in self.graph:
            raise KeyError(f"unknown topic '{topic}'")
        when = when or time.time()
        row = self.db.execute(
            "SELECT times FROM coverage WHERE topic=?", (topic,)).fetchone()
        times = (row[0] if row else 0) + 1
        self.db.execute("INSERT OR REPLACE INTO coverage VALUES (?,?,?,?)",
                        (topic, times, when, video_id))
        self.db.commit()

    def coverage(self) -> Dict[str, Dict[str, Any]]:
        return {t: {"times": n, "last_taught": ts, "last_video": vid}
                for t, n, ts, vid in self.db.execute(
                    "SELECT topic, times, last_taught, last_video "
                    "FROM coverage").fetchall()}

    def covered(self) -> Set[str]:
        return set(self.coverage())

    def ready(self, topic: str) -> bool:
        """Every prerequisite already taught at least once."""
        done = self.covered()
        return all(p in done for p in self.graph[topic].get("prereqs", []))

    def due_at(self, topic: str) -> Optional[float]:
        """When this topic's next spaced revisit becomes due (epoch), or
        None if it is exhausted (past the ladder)."""
        cov = self.coverage().get(topic)
        if not cov or not cov.get("last_taught"):
            return 0.0                              # never taught → due now
        n = int(cov["times"])
        if n >= len(REVISIT_LADDER_DAYS):
            return None
        return float(cov["last_taught"]) + REVISIT_LADDER_DAYS[n] * 86400.0

    # ─── the scheduler ─────────────────────────────────────

    def plan_next(self, demand_queue: Optional[Sequence[Dict[str, Any]]] = None,
                  limit: int = 10, now: Optional[float] = None
                  ) -> List[Dict[str, Any]]:
        """Ranked, teachable queue.

        score = demand (audience evidence) + readiness + due-ness, with
        NEW topics preferred over revisits at equal demand and topics
        whose prerequisites are missing excluded outright.
        """
        now = now or time.time()
        demand: Dict[str, float] = {}
        questions: Dict[str, str] = {}
        for rec in demand_queue or []:
            tid = rec.get("topic")
            if not tid or tid not in self.graph:
                continue
            demand[tid] = max(demand.get(tid, 0.0),
                              float(rec.get("demand", 0.0))
                              * (0.5 + float(rec.get("confidence", 0.0))))
            questions.setdefault(tid, str(rec.get("question", "")))

        order = {t: i for i, t in enumerate(topological_order(self.graph))}
        cov = self.coverage()
        out: List[Dict[str, Any]] = []
        for tid, node in self.graph.items():
            if not self.ready(tid):
                continue
            due = self.due_at(tid)
            if due is None:
                continue
            if due > now:
                continue
            times = int(cov.get(tid, {}).get("times", 0))
            novelty = 1.6 if times == 0 else 0.9 ** times
            overdue_days = max(0.0, (now - due) / 86400.0) if times else 0.0
            score = (demand.get(tid, 0.0) * 1.4
                     + novelty
                     + min(1.5, overdue_days / 14.0)
                     - order[tid] * 0.004)          # gentle syllabus order bias
            out.append({
                "topic": tid,
                "title": node["title"],
                "subject": node["subject"],
                "chapter": node["chapter"],
                "state": "new" if times == 0 else f"revisit#{times + 1}",
                "demand": round(demand.get(tid, 0.0), 3),
                "score": round(score, 3),
                "audience_question": questions.get(tid, ""),
                "prerequisites": list(node.get("prereqs", [])),
            })
        out.sort(key=lambda r: (-r["score"], r["topic"]))
        return out[:limit]

    def plan_from_comments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """The full loop: mined doubts → demand → teachable queue."""
        try:
            from agents.comment_miner import CommentMiner
            queue = CommentMiner().topic_queue(limit=60)
        except Exception as e:                      # noqa: BLE001 — optional
            print(f"  [Syllabus] comment demand unavailable: {e}")
            queue = []
        return self.plan_next(queue, limit=limit)

    # ─── reporting ─────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        cov = self.coverage()
        total = len(self.graph)
        return {
            "topics": total,
            "covered": len(cov),
            "coverage_pct": round(100.0 * len(cov) / max(1, total), 1),
            "blocked": [t for t in self.graph if not self.ready(t)],
            "next": self.plan_next(limit=8),
        }

    def describe(self) -> str:
        r = self.report()
        lines = [f"═══ Syllabus · {r['covered']}/{r['topics']} topics "
                 f"({r['coverage_pct']}%) ═══"]
        for item in r["next"]:
            lines.append(f"  [{item['score']:.2f}] {item['subject']:<9} "
                         f"{item['title']:<38} {item['state']}")
            if item["audience_question"]:
                lines.append(f"      ← “{item['audience_question'][:80]}”")
        if r["blocked"]:
            lines.append(f"  blocked on prerequisites: {len(r['blocked'])}")
        return "\n".join(lines)


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="syllabus coverage + scheduler")
    ap.add_argument("--mark", nargs=2, metavar=("TOPIC", "VIDEO_ID"))
    ap.add_argument("--order", action="store_true",
                    help="print the prerequisite-first teaching order")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    syl = Syllabus()
    if args.mark:
        syl.mark_taught(args.mark[0], args.mark[1])
    if args.order:
        for i, t in enumerate(topological_order(), 1):
            print(f"  {i:>2}. {t}")
        return 0
    if args.json:
        print(json.dumps(syl.report(), indent=2, ensure_ascii=False))
    else:
        print(syl.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
