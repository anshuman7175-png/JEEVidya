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
from pipeline.script_parser import parse_dialogue_script, split_audio_by_silence

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


UPLOAD_AUDIO_DIR = os.path.join(settings.PROJECT_ROOT, "uploads", "audio")
os.makedirs(UPLOAD_AUDIO_DIR, exist_ok=True)


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Upload custom MP3/WAV/WebM voiceover for a specific dialogue turn."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if not file or not file.filename:
        return jsonify({"error": "Empty audio file"}), 400

    turn_id = request.form.get('turn_id', '0')
    ext = os.path.splitext(file.filename)[1].lower() or '.webm'
    ts = int(time.time() * 1000)
    raw_path = os.path.join(UPLOAD_AUDIO_DIR, f"raw_turn_{turn_id}_{ts}{ext}")
    final_mp3 = os.path.join(UPLOAD_AUDIO_DIR, f"turn_{turn_id}_{ts}.mp3")

    file.save(raw_path)

    # Transcode to standardized 44.1kHz stereo MP3
    import subprocess
    from pipeline.voice import _get_audio_duration_ms
    try:
        cmd = ["ffmpeg", "-y", "-i", raw_path, "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", final_mp3]
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(raw_path) and raw_path != final_mp3:
            try:
                os.remove(raw_path)
            except OSError:
                pass
    except Exception as e:
        if ext == '.mp3':
            os.replace(raw_path, final_mp3)
        else:
            return jsonify({"error": f"Failed to process audio: {e}"}), 500

    duration_ms = _get_audio_duration_ms(final_mp3)
    duration_s = round(duration_ms / 1000.0, 2)
    filename = os.path.basename(final_mp3)

    return jsonify({
        "success": True,
        "path": final_mp3,
        "filename": filename,
        "url": f"/audio/{filename}",
        "duration_ms": duration_ms,
        "duration_s": duration_s,
    })


@app.route('/parse_script', methods=['POST'])
def parse_script_endpoint():
    """Parse raw multi-line script into structured dialogue turns with continuity detection."""
    data = request.json or {}
    raw_script = data.get("script", "")
    group_consecutive = data.get("group_consecutive", False)
    default_first_speaker = data.get("default_first_speaker", "girl")

    try:
        turns = parse_dialogue_script(
            raw_script=raw_script,
            group_consecutive=group_consecutive,
            default_first_speaker=default_first_speaker,
        )
        return jsonify({
            "success": True,
            "turns": turns,
            "turn_count": len(turns),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/split_full_audio', methods=['POST'])
def split_full_audio_endpoint():
    """Split a single full voiceover track into per-turn MP3 files using silence detection."""
    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    file = request.files['audio']
    if not file or not file.filename:
        return jsonify({"success": False, "error": "Empty audio file"}), 400

    expected_turns = request.form.get('expected_turns')
    if expected_turns:
        try:
            expected_turns = int(expected_turns)
        except ValueError:
            expected_turns = None

    min_silence_len = int(request.form.get('min_silence_len', 280))
    silence_thresh_db = float(request.form.get('silence_thresh_db', -38.0))

    ts = int(time.time() * 1000)
    ext = os.path.splitext(file.filename)[1].lower() or '.mp3'
    input_path = os.path.join(UPLOAD_AUDIO_DIR, f"full_{ts}{ext}")
    file.save(input_path)

    slices_subdir = f"slices_{ts}"
    slices_dir = os.path.join(UPLOAD_AUDIO_DIR, slices_subdir)

    try:
        slices = split_audio_by_silence(
            input_audio_path=input_path,
            output_dir=slices_dir,
            expected_turns=expected_turns,
            min_silence_len=min_silence_len,
            silence_thresh_db=silence_thresh_db,
        )
        for s in slices:
            rel_file = f"{slices_subdir}/{s['filename']}"
            s["url"] = f"/audio/{rel_file}"

        return jsonify({
            "success": True,
            "slices": slices,
            "slice_count": len(slices),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(UPLOAD_AUDIO_DIR, filename)


@app.route('/load_last_session', methods=['GET'])
def load_last_session():
    candidates = [
        os.path.join(settings.PROJECT_ROOT, "scenes", "last_session.json"),
        os.path.join(settings.PROJECT_ROOT, "scenes", "venus.json"),
        os.path.join(settings.PROJECT_ROOT, "scenes", "aasman_blue_kyu_hai.json")
    ]
    last_session_path = None
    for cand in candidates:
        if os.path.exists(cand):
            last_session_path = cand
            break
    if not last_session_path:
        return jsonify({"error": "No saved session found"}), 404
    try:
        with open(last_session_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        last_video = data.get("output_filename")
        if not last_video:
            # Fallback: scan output directory for newest completed mp4
            mp4s = [f for f in os.listdir(settings.OUTPUT_DIR) if f.endswith(".mp4") and "preview" not in f and "perfected" not in f]
            if mp4s:
                mp4s.sort(key=lambda x: os.path.getmtime(os.path.join(settings.OUTPUT_DIR, x)), reverse=True)
                last_video = mp4s[0]
        return jsonify({"success": True, "dialogue": data, "last_video": last_video})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/scenes', methods=['GET'])
def list_scenes():
    """List available scene presets (Season 1 and Season 2)."""
    try:
        scenes_dir = os.path.join(settings.PROJECT_ROOT, "scenes")
        season2_index_file = os.path.join(scenes_dir, "season2_index.json")
        season2_data = {}
        if os.path.exists(season2_index_file):
            with open(season2_index_file, "r", encoding="utf-8") as f:
                season2_data = json.load(f)

        season1_scenes = []
        for fn in ["aasman_blue_kyu_hai.json", "example.json"]:
            fp = os.path.join(scenes_dir, fn)
            if os.path.exists(fp):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        sc = json.load(f)
                    season1_scenes.append({
                        "id": fn.replace(".json", ""),
                        "title": sc.get("title", fn),
                        "description": sc.get("description", ""),
                        "turn_count": len(sc.get("turns", [])),
                        "filename": fn
                    })
                except Exception:
                    pass

        return jsonify({
            "success": True,
            "season1": season1_scenes,
            "season2": season2_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/scenes/<scene_id>', methods=['GET'])
def get_scene(scene_id):
    """Retrieve full scene data by ID or filename."""
    try:
        scenes_dir = os.path.join(settings.PROJECT_ROOT, "scenes")
        season2_dir = os.path.join(scenes_dir, "season2")

        target_path = None
        candidates = [
            os.path.join(scenes_dir, f"{scene_id}.json"),
            os.path.join(scenes_dir, scene_id),
            os.path.join(season2_dir, f"{scene_id}.json"),
            os.path.join(season2_dir, scene_id)
        ]
        if os.path.exists(season2_dir):
            for fn in os.listdir(season2_dir):
                if fn.startswith(f"{scene_id}_") or fn.startswith(f"{scene_id}."):
                    candidates.append(os.path.join(season2_dir, fn))

        for cand in candidates:
            if os.path.exists(cand) and os.path.isfile(cand):
                target_path = cand
                break

        if not target_path:
            return jsonify({"success": False, "error": f"Scene '{scene_id}' not found"}), 404

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return jsonify({"success": True, "scene": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate_video():
    global _generating, _progress

    if _generating:
        return jsonify({"error": "A video is already being generated. Please wait."}), 429

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Save session to scenes/last_session.json immediately
        scenes_dir = os.path.join(settings.PROJECT_ROOT, "scenes")
        os.makedirs(scenes_dir, exist_ok=True)
        session_file = os.path.join(scenes_dir, "last_session.json")
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as se:
            print(f"  [Session] Warning: failed to write last_session.json: {se}")

        _generating = True
        _progress = {"stage": "INIT", "percent": 0, "message": "Starting..."}

        def run_in_thread():
            global _generating, _progress
            try:
                output_path = run_dialogue_pipeline(data, progress_callback=_progress_callback)
                output_filename = os.path.basename(output_path)
                # Update session with completed output filename
                try:
                    data["output_filename"] = output_filename
                    with open(session_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

                _progress = {
                    "stage": "DONE",
                    "percent": 100,
                    "message": f"Video saved: {output_filename}",
                    "output_path": output_path,
                    "output_filename": output_filename,
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
