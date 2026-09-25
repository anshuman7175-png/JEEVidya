"""
Tests for Procedural Physics Simulations (§6).
"""
import pytest
from PIL import Image

from engine.sims import (
    OrbitSimulation,
    ElevatorSimulation,
    NewtonMountainSimulation,
    EscapeVelocitySimulation,
    get_simulation,
)


def test_orbit_simulation_render():
    sim = OrbitSimulation(mode="circular")
    frame = sim.render(t_seconds=1.5, params={"mode": "circular"}, width=1080, height=820, res_scale=1.0)
    assert isinstance(frame, Image.Image)
    assert frame.size == (1080, 820)
    assert frame.mode == "RGBA"


def test_orbit_moon_mode_render():
    sim = OrbitSimulation(mode="moon_orbit")
    frame = sim.render(t_seconds=2.0, params={"mode": "moon_orbit", "target": "moon"}, width=1080, height=820, res_scale=1.0)
    assert isinstance(frame, Image.Image)
    assert frame.size == (1080, 820)
    assert frame.mode == "RGBA"


def test_newton_mountain_simulation_render():
    sim = NewtonMountainSimulation(mode="drop")
    for mode in ["drop", "throw_short", "throw_far", "orbital_loop"]:
        frame = sim.render(t_seconds=1.2, params={"mode": mode}, width=1080, height=820, res_scale=1.0)
        assert isinstance(frame, Image.Image)
        assert frame.size == (1080, 820)
        assert frame.mode == "RGBA"


def test_elevator_simulation_render():
    sim = ElevatorSimulation(mode="static")
    frame_static = sim.render(t_seconds=0.5, params={"mode": "static"}, width=1080, height=820, res_scale=1.0)
    assert isinstance(frame_static, Image.Image)
    assert frame_static.size == (1080, 820)

    # Free fall cable snap
    frame_snap = sim.render(t_seconds=3.0, params={"mode": "cable_snap", "snap_time": 1.5}, width=1080, height=820, res_scale=1.0)
    assert isinstance(frame_snap, Image.Image)
    assert frame_snap.size == (1080, 820)


def test_simulation_registry():
    sim_orbit = get_simulation("orbit")
    assert isinstance(sim_orbit, OrbitSimulation)

    sim_elevator = get_simulation("elevator")
    assert isinstance(sim_elevator, ElevatorSimulation)

    sim_mountain = get_simulation("newton_mountain")
    assert isinstance(sim_mountain, NewtonMountainSimulation)

    sim_cannon = get_simulation("newton_cannon")
    assert isinstance(sim_cannon, NewtonMountainSimulation)

    sim_none = get_simulation("non_existent_simulation")
    assert sim_none is None


def test_sim_resolution_scaling():
    sim = OrbitSimulation()
    frame_half = sim.render(t_seconds=0.0, params={}, width=540, height=410, res_scale=0.5)
    assert frame_half.size == (270, 205) or frame_half.size == (540, 410)

    sim_mountain = NewtonMountainSimulation()
    frame_nm_half = sim_mountain.render(t_seconds=0.5, params={"mode": "throw_far"}, width=540, height=410, res_scale=0.5)
    assert frame_nm_half.size == (540, 410)


def test_escape_velocity_simulation_render():
    sim = EscapeVelocitySimulation()
    for mode in ["earth_escape", "gravity_well_comparison", "atmospheric_escape"]:
        # Test start and mid-flight
        for t in [0.0, 1.5, 3.5]:
            frame = sim.render(t_seconds=t, params={"mode": mode}, width=1080, height=820, res_scale=1.0)
            assert isinstance(frame, Image.Image)
            assert frame.size == (1080, 820)
            assert frame.mode == "RGBA"

    # Test half-scale resolution
    frame_half = sim.render(t_seconds=1.0, params={"mode": "earth_escape"}, width=540, height=410, res_scale=0.5)
    assert frame_half.size == (540, 410)
    assert frame_half.mode == "RGBA"


def test_escape_velocity_registry():
    for alias in ["escape_velocity", "escape", "atmospheric_escape", "moon_atmosphere"]:
        sim = get_simulation(alias)
        assert isinstance(sim, EscapeVelocitySimulation)


