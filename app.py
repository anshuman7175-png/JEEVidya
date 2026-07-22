"""
Gudiya & Chintu — Web Interface V2
Visual dialogue editor with real-time progress streaming.
"""
import json
import os
import threading
import time

from flask import Flask, Response, render_template, request, jsonify, send_from_directory

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate import run_dialogue_pipeline
from config import settings
from config.prompts import EXAMPLE_DIALOGUE

app = Flask(__name__)

# Global progress state
_progress = {"stage": "IDLE", "percent": 0, "message": "Ready"}
_generating = False


def _progress_callback(stage: str, pct: float, msg: str):
    global _progress
    _progress = {"stage": stage, "percent": pct, "message": msg}


@app.route('/')
def index():
    return render_template('index.html',
                           default_json=json.dumps(EXAMPLE_DIALOGUE, indent=2, ensure_ascii=False))


@app.route('/generate', methods=['POST'])
def generate_video():
    global _generating, _progress

    if _generating:
        return jsonify({"error": "A video is already being generated. Please wait."}), 429

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        _generating = True
        _progress = {"stage": "INIT", "percent": 0, "message": "Starting..."}

        def run_in_thread():
            global _generating, _progress
            try:
                output_path = run_dialogue_pipeline(data, progress_callback=_progress_callback)
                _progress = {
                    "stage": "DONE",
                    "percent": 100,
                    "message": f"Video saved: {os.path.basename(output_path)}",
                    "output_path": output_path,
                    "output_filename": os.path.basename(output_path),
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                _progress = {"stage": "ERROR", "percent": 0, "message": str(e)}
            finally:
                _generating = False

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        return jsonify({"success": True, "message": "Generation started. Check /status for progress."})

    except Exception as e:
        _generating = False
        return jsonify({"error": str(e)}), 500


@app.route('/status')
def status():
    """SSE endpoint for real-time progress streaming."""
    def generate():
        last_msg = ""
        while True:
            if _progress["message"] != last_msg:
                last_msg = _progress["message"]
                yield f"data: {json.dumps(_progress)}\n\n"
            if _progress["stage"] in ("DONE", "ERROR"):
                yield f"data: {json.dumps(_progress)}\n\n"
                break
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/output/<filename>')
def serve_output(filename):
    """Serve generated videos for in-browser playback."""
    return send_from_directory(settings.OUTPUT_DIR, filename)


# ═══════════════════════════════════════════
# TIER 1 — /studio : puppet rig editor
# ═══════════════════════════════════════════

_studio_engines = {}          # character → BoneEngine (invalidated on save)


def _studio_engine(character: str):
    from engine.rig import Rig
    from engine.bone_engine import BoneEngine
    if character not in _studio_engines:
        _studio_engines[character] = BoneEngine(Rig.load(character))
    return _studio_engines[character]


def _characters():
    out = []
    for name in sorted(os.listdir(settings.CHARACTERS_DIR)):
        d = os.path.join(settings.CHARACTERS_DIR, name)
        if os.path.isdir(d) and any(
                os.path.exists(os.path.join(d, f"body{e}"))
                for e in (".png", ".jpg", ".jpeg")):
            out.append(name)
    return out


@app.route('/studio')
def studio():
    return render_template('studio.html', characters=_characters())


@app.route('/studio/data/<character>')
def studio_data(character):
    from engine.rig import Rig, rig_path
    if character not in _characters():
        return jsonify({"error": "unknown character"}), 404
    if not os.path.exists(rig_path(character)):
        return jsonify({"rigged": False})
    rig = Rig.load(character)
    d = rig.to_dict()
    d["rigged"] = True
    return jsonify(d)


@app.route('/studio/image/<character>')
def studio_image(character):
    char_dir = os.path.join(settings.CHARACTERS_DIR, character)
    for ext in ('.png', '.jpg', '.jpeg'):
        if os.path.exists(os.path.join(char_dir, f"body{ext}")):
            return send_from_directory(char_dir, f"body{ext}")
    return "not found", 404


@app.route('/studio/preview/<character>')
def studio_preview(character):
    """Live Bone Engine render of an arbitrary test pose."""
    import io
    from flask import request as req
    from engine.bone_engine import PuppetPose

    def f(name, default=0.0):
        try:
            return float(req.args.get(name, default))
        except (TypeError, ValueError):
            return default

    pose = PuppetPose(
        lean=f("lean"), head_tilt=f("tilt"), head_yaw=f("yaw"),
        head_nod=f("nod"), squash=f("squash"),
        viseme=req.args.get("viseme", "REST"),
        mouth_open=f("open"), blink=f("blink"), brow=f("brow"))
    try:
        img = _studio_engine(character).render(pose)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # Downscale for snappy previews
    if img.height > 720:
        img = img.resize((int(img.width * 720 / img.height), 720))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return Response(buf.read(), mimetype="image/png")


@app.route('/studio/save/<character>', methods=['POST'])
def studio_save(character):
    """Nudged joints/face boxes → rig.json → re-slice + re-bake sprites."""
    from engine.rig import Rig
    from tools.rig_builder import rebake
    try:
        data = request.json or {}
        rig = Rig.load(character)
        for k, v in (data.get("joints") or {}).items():
            rig.joints[k] = (float(v[0]), float(v[1]))
        for k, v in (data.get("face") or {}).items():
            if k in ("skin", "lip"):
                rig.face[k] = tuple(int(c) for c in v)
            else:
                rig.face[k] = tuple(float(c) for c in v)
        for k, v in (data.get("params") or {}).items():
            rig.params[k] = float(v)
        rig.generated_by = "manual"
        rebake(rig)
        _studio_engines.pop(character, None)
        return jsonify({"success": True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/studio/build/<character>', methods=['POST'])
def studio_build(character):
    """Full automatic re-rig (mediapipe detection from scratch)."""
    from tools.rig_builder import build_rig
    try:
        rig = build_rig(character, force=True)
        _studio_engines.pop(character, None)
        return jsonify({"success": True, "generated_by": rig.generated_by})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Gudiya & Chintu — Animated Shorts Factory")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
