import os
import json
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_homepage_has_season2_dropdown(client):
    rv = client.get("/")
    assert rv.status_code == 200
    html = rv.get_data(as_text=True)
    assert 'id="season2-preset-dropdown"' in html
    assert 'id="modal-s2-selector"' in html
    assert 'initSeason2Presets' in html

def test_api_scenes_list(client):
    rv = client.get("/api/scenes")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    s2 = data.get("season2", {})
    assert s2.get("total_episodes") == 20
    episodes = s2.get("episodes", [])
    assert len(episodes) == 20
    assert episodes[0]["id"] == "s02e01"
    assert episodes[0]["title"] == "Astronaut float kyun karte hain?"
    assert episodes[19]["id"] == "s02e20"

def test_all_20_episodes_loadable(client):
    for ep_num in range(1, 21):
        scene_id = f"s02e{ep_num:02d}"
        rv = client.get(f"/api/scenes/{scene_id}")
        assert rv.status_code == 200, f"Episode {scene_id} failed to load"
        data = rv.get_json()
        assert data["success"] is True
        sc = data["scene"]
        assert sc["season"] == 2
        assert sc["episode"] == ep_num
        assert len(sc["title"]) > 0
        assert len(sc["hook_text"]) > 0
        assert len(sc["ncert_chapter"]) > 0
        assert len(sc["concept"]) > 0
        assert len(sc["turns"]) >= 12
        for t in sc["turns"]:
            assert t["speaker"] in ("girl", "boy")
            assert len(t["text"].strip()) > 0
