"""Test custom MP3 ingestion, alignment, and VTT generation."""
import os
import subprocess
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.voice import VoiceEngine
from pipeline.timeline import Timeline, parse_vtt_words


@pytest.fixture
def sample_mp3(tmp_path):
    """Generate a clean 2-second test MP3 file using ffmpeg."""
    mp3_path = str(tmp_path / "sample_voice.mp3")
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2.0",
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        mp3_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return mp3_path


def test_custom_audio_turn_processing(sample_mp3, tmp_path):
    """VoiceEngine must process custom audio, extract duration, and emit VTT cues."""
    engine = VoiceEngine()
    out_dir = str(tmp_path / "voice_out")

    text = "Yeh ek parikshan vakya hai"
    res = engine.generate_turn_audio(
        turn_id=1,
        text=text,
        speaker="girl",
        output_dir=out_dir,
        custom_audio=sample_mp3
    )

    assert res["audio"] is not None
    assert os.path.exists(res["audio"])
    assert res["vtt"] is not None
    assert os.path.exists(res["vtt"])
    assert res["duration_ms"] >= 1900  # ~2000 ms

    # Check that VTT has parsed word events
    words = parse_vtt_words(res["vtt"])
    assert len(words) >= 4
    assert words[0].text == "Yeh"
    assert words[-1].end_ms <= res["duration_ms"] + 100


def test_custom_audio_in_dialogue_pipeline(sample_mp3, tmp_path):
    """generate_dialogue must process turns with custom_audio correctly."""
    engine = VoiceEngine()
    out_dir = str(tmp_path / "dialogue_out")

    dialogue = {
        "title": "Custom Test",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "girl",
                "text": "Namaste dosto",
                "custom_audio": sample_mp3,
                "emotion": "enthusiastic",
                "shot_type": "two_shot",
            },
            {
                "turn_id": 2,
                "speaker": "explanation",
                "duration_seconds": 2.0,
                "pause_after": 0.5,
            }
        ]
    }

    turn_data = engine.generate_dialogue(dialogue, out_dir)
    assert len(turn_data) == 2
    assert turn_data[0]["speaker"] == "girl"
    assert turn_data[0]["audio"] is not None
    assert turn_data[0]["vtt"] is not None
    assert turn_data[0]["duration_ms"] > turn_data[0]["audio_duration_ms"]  # includes padding

    # Verify Timeline consumes this without drift
    timeline = Timeline(turn_data, fps=30)
    assert timeline.total_frames == round(timeline.total_ms * 30 / 1000)
    assert len(timeline.spans) == 2


def test_leading_number_preserved(sample_mp3, tmp_path):
    """Lines starting with numbers (e.g. '90 percent?!') must NOT strip the number."""
    engine = VoiceEngine()
    out_dir = str(tmp_path / "voice_out_num")
    text = "90 percent?! Toh woh float kaise kar rahe hain?"
    res = engine.generate_turn_audio(
        turn_id=5,
        text=text,
        speaker="girl",
        output_dir=out_dir,
        custom_audio=sample_mp3,
    )
    words = parse_vtt_words(res["vtt"])
    assert len(words) >= 8
    assert words[0].text == "90"
