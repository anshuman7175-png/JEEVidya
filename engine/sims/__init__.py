"""
JEEVidya — Procedural Physics Simulations Library
══════════════════════════════════════════════════
Provides high-fidelity, resolution-independent 60fps procedural animations
for the Visual Bible Lab canvas.
"""
from typing import Any, Dict, Optional

from engine.sims.elevator import ElevatorSimulation
from engine.sims.escape_velocity import EscapeVelocitySimulation
from engine.sims.newton_mountain import NewtonMountainSimulation
from engine.sims.orbit import OrbitSimulation

SIM_REGISTRY = {
    "orbit": OrbitSimulation,
    "iss_orbit": OrbitSimulation,
    "orbital_freefall": OrbitSimulation,
    "elevator": ElevatorSimulation,
    "free_fall_elevator": ElevatorSimulation,
    "newton_mountain": NewtonMountainSimulation,
    "newton_cannon": NewtonMountainSimulation,
    "mountain": NewtonMountainSimulation,
    "escape_velocity": EscapeVelocitySimulation,
    "escape": EscapeVelocitySimulation,
    "atmospheric_escape": EscapeVelocitySimulation,
    "moon_atmosphere": EscapeVelocitySimulation,
}


def get_simulation(name: str, **kwargs: Any) -> Optional[Any]:
    """Retrieve an initialized simulation instance by registered name."""
    norm_name = name.strip().lower()
    sim_cls = SIM_REGISTRY.get(norm_name)
    if sim_cls:
        return sim_cls(**kwargs)
    return None


__all__ = [
    "OrbitSimulation",
    "ElevatorSimulation",
    "NewtonMountainSimulation",
    "EscapeVelocitySimulation",
    "get_simulation",
    "SIM_REGISTRY",
]

