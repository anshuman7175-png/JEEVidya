"""
Unit tests for pipeline/script_parser.py
Tests whole script parsing, speaker continuity detection, emotion parsing,
and audio silence segmentation.
"""
import os
import shutil
import tempfile
import pytest
import pydub
from pydub.generators import Sine

from pipeline.script_parser import (
    parse_dialogue_script,
    split_audio_by_silence,
    normalize_speaker,
    extract_emotion_and_shot,
)


def test_speaker_normalization():
    assert normalize_speaker("Gudiya") == "girl"
    assert normalize_speaker("guddi") == "girl"
    assert normalize_speaker("Didi") == "girl"
    assert normalize_speaker("G") == "girl"
    assert normalize_speaker("गुड़िया") == "girl"
    assert normalize_speaker("Chintu") == "boy"
    assert normalize_speaker("bhai") == "boy"
    assert normalize_speaker("C") == "boy"
    assert normalize_speaker("चिंटू") == "boy"
    assert normalize_speaker("Slide") == "explanation"
    assert normalize_speaker("Blackboard") == "explanation"


def test_extract_emotion_and_shot():
    em, shot, clean = extract_emotion_and_shot("[curious] Kya tare timtimate hain?", "girl")
    assert em == "curious"
    assert shot == "two_shot"
    assert clean == "Kya tare timtimate hain?"

    em2, shot2, clean2 = extract_emotion_and_shot("(amazed, medium) Arre waah!", "boy")
    assert em2 == "amazed"
    assert shot2 == "medium"
    assert clean2 == "Arre waah!"


def test_parse_multi_line_consecutive_script():
    raw = """
    Gudiya: Chintu, suno ek baat.
    Physics bohot interesting subject hai agar dhyan se samjho toh.
    Chintu: Sahi keh rahi hain aap Didi.
    [Slide: v_e = sqrt(2GM/R)]
    Gudiya: [amazed] Aur yeh formula toh sabse aasan hai!
    """
    turns = parse_dialogue_script(raw, group_consecutive=False)

    # Turns:
    # 1: Gudiya line 1 of 2
    # 2: Gudiya line 2 of 2
    # 3: Chintu line 1 of 1
    # 4: Slide (explanation)
    # 5: Gudiya line 1 of 1
    assert len(turns) == 5

    # Check Gudiya consecutive lines
    assert turns[0]["speaker"] == "girl"
    assert turns[0]["line_in_block"] == 1
    assert turns[0]["total_in_block"] == 2
    assert turns[0]["is_block_start"] is True
    assert turns[0]["is_block_end"] is False

    assert turns[1]["speaker"] == "girl"
    assert turns[1]["line_in_block"] == 2
    assert turns[1]["total_in_block"] == 2
    assert turns[1]["is_block_start"] is False
    assert turns[1]["is_block_end"] is True

    # Check Chintu
    assert turns[2]["speaker"] == "boy"
    assert turns[2]["total_in_block"] == 1

    # Check Slide
    assert turns[3]["speaker"] == "explanation"
    assert "v_e = sqrt(2GM/R)" in turns[3]["text"]

    # Check Gudiya emotion
    assert turns[4]["speaker"] == "girl"
    assert turns[4]["emotion"] == "amazed"


def test_parse_group_consecutive():
    raw = """
    Gudiya: Line 1.
    Line 2.
    Line 3.
    Chintu: Reply 1.
    Reply 2.
    """
    turns = parse_dialogue_script(raw, group_consecutive=True)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "girl"
    assert turns[0]["consecutive_lines_count"] == 3
    assert turns[0]["text"] == "Line 1. Line 2. Line 3."

    assert turns[1]["speaker"] == "boy"
    assert turns[1]["consecutive_lines_count"] == 2
    assert turns[1]["text"] == "Reply 1. Reply 2."


def test_parse_unlabeled_alternating_dialogue():
    raw = """
    First line by Gudiya
    Second line by Chintu
    Third line by Gudiya
    Fourth line by Chintu
    """
    turns = parse_dialogue_script(raw, default_first_speaker="girl")
    assert len(turns) == 4
    assert turns[0]["speaker"] == "girl"
    assert turns[1]["speaker"] == "boy"
    assert turns[2]["speaker"] == "girl"
    assert turns[3]["speaker"] == "boy"


def test_split_audio_by_silence():
    temp_dir = tempfile.mkdtemp(prefix="test_silence_")
    try:
        # Generate 3 audio pulses separated by 400ms silence
        tone1 = Sine(440).to_audio_segment(duration=600)
        silence = pydub.AudioSegment.silent(duration=400)
        tone2 = Sine(880).to_audio_segment(duration=700)
        tone3 = Sine(660).to_audio_segment(duration=500)

        full_audio = tone1 + silence + tone2 + silence + tone3
        input_mp3 = os.path.join(temp_dir, "full_input.mp3")
        full_audio.export(input_mp3, format="mp3")

        output_dir = os.path.join(temp_dir, "slices")
        slices = split_audio_by_silence(
            input_audio_path=input_mp3,
            output_dir=output_dir,
            expected_turns=3,
            min_silence_len=250,
            silence_thresh_db=-30.0,
        )

        assert len(slices) == 3
        for s in slices:
            assert os.path.exists(s["path"])
            assert s["duration_ms"] > 400
            assert s["filename"].startswith("turn_")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_flask_parse_script_endpoint():
    from app import app
    client = app.test_client()

    payload = {
        "script": "Gudiya: Pehla sawal.\nChintu: Dusra jawab.",
        "group_consecutive": False,
        "default_first_speaker": "girl"
    }
    res = client.post('/parse_script', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["turn_count"] == 2
    assert data["turns"][0]["speaker"] == "girl"
    assert data["turns"][1]["speaker"] == "boy"


def test_flask_split_full_audio_endpoint():
    from app import app
    import io
    client = app.test_client()

    tone1 = Sine(440).to_audio_segment(duration=500)
    silence = pydub.AudioSegment.silent(duration=400)
    tone2 = Sine(880).to_audio_segment(duration=600)
    full = tone1 + silence + tone2

    buf = io.BytesIO()
    full.export(buf, format="mp3")
    buf.seek(0)

    data = {
        'audio': (buf, 'full_audio.mp3'),
        'expected_turns': '2',
        'min_silence_len': '250',
        'silence_thresh_db': '-30.0'
    }
    res = client.post('/split_full_audio', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_data = res.get_json()
    assert res_data["success"] is True
    assert res_data["slice_count"] == 2
    for s in res_data["slices"]:
        assert "url" in s
        assert "duration_s" in s


def test_parse_speaker_labels_and_strip_metadata():
    raw_script = """
    Title: Venus Ulta Kyu Ghumta Hai?
    Description: Ek bohot hi rochak factual video.

    Speaker 1: Kya aapko pata hai ki Venus baki planets se ulta ghumta hai?
    Speaker 2: Sach me Didi? Par aisa kyu hota hai?
    **Speaker 1:** Scientist maante hain ki ek bada asteroid takraya tha.
    [Chintu]: Arre baap re! Yeh to bohot khatarnak hai!

    Tags: #space #venus #solarsystem #science
    #shorts #viral #facts
    """
    turns = parse_dialogue_script(raw_script, group_consecutive=False)
    assert len(turns) == 4

    # Verify speaker assignment
    assert turns[0]["speaker"] == "girl"
    assert turns[1]["speaker"] == "boy"
    assert turns[2]["speaker"] == "girl"
    assert turns[3]["speaker"] == "boy"

    # Verify no speaker labels, markdown, or tags remain in the spoken dialogue text
    for t in turns:
        txt = t["text"].lower()
        assert not txt.startswith("speaker 1")
        assert not txt.startswith("speaker 2")
        assert not txt.startswith("gudiya")
        assert not txt.startswith("chintu")
        assert "tags" not in txt
        assert "title" not in txt
        assert "#" not in txt

    assert turns[0]["text"] == "Kya aapko pata hai ki Venus baki planets se ulta ghumta hai?"
    assert turns[1]["text"] == "Sach me Didi? Par aisa kyu hota hai?"
    assert turns[2]["text"] == "Scientist maante hain ki ek bada asteroid takraya tha."
    assert turns[3]["text"] == "Arre baap re! Yeh to bohot khatarnak hai!"


def test_sanitize_spoken_text_defense():
    from pipeline.voice import sanitize_spoken_text

    assert sanitize_spoken_text("Tags: #space #venus") == ""
    assert sanitize_spoken_text("#shorts #trending #facts") == ""
    assert sanitize_spoken_text("Title: Venus Mystery") == ""
    assert sanitize_spoken_text("Speaker 1: Namaste dosto") == "Namaste dosto"
    assert sanitize_spoken_text("**Speaker 2:** Haan bilkul!") == "Haan bilkul!"
    assert sanitize_spoken_text("[Gudiya]: Sahi kaha!") == "Sahi kaha!"
    assert sanitize_spoken_text("Gudiya (curious): Kya tare timtimate hain?") == "Kya tare timtimate hain?"
    assert sanitize_spoken_text("Turn 1: Yeh to gazab hai!") == "Yeh to gazab hai!"
    assert sanitize_spoken_text('"Double quoted line"') == "Double quoted line"

