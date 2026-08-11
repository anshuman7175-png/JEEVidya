# Voice Recording Guide (Terminal Plan, Part X §10.5)

Record **once**, never per line. Two kinds of clips per character:

| Clip | Length | Content |
|------|--------|---------|
| `base.wav` | 45–60 s | Neutral, even-paced read of the prompt sheet below |
| `emotion/<name>.wav` | 5–10 s each | The same 1–2 sentences read *in that emotion* |

Files go under `assets/voices/<character>/` and are registered in
`assets/voices/voice_bank.json`. Validate before use:

```bash
python3 -c "from pipeline.voice_clone import load_voice_bank; load_voice_bank(); print('voice bank OK')"
```

The validator hard-rejects clipping, high noise floor, DC offset, and
low sample rate — bad reference audio would poison **every** downstream
line, so it is refused at the door.

## Recording setup

- **Mic:** any decent USB condenser ≥ 44.1 kHz / 16-bit. Phone voice-memo
  quality is acceptable *if* the room is quiet.
- **Room:** soft furnishings, no fan/AC audible, no echo (a closet full of
  clothes beats an empty room).
- **Levels:** peak between −12 and −6 dBFS. Never touch 0 dBFS (the
  validator rejects clipping). Keep 15–20 cm mic distance, slight
  off-axis to kill plosives.
- **Format:** WAV, mono, 44.1 or 48 kHz. Do not send MP3.
- **Consistency:** record ALL clips in one sitting — same mic, same
  distance, same room. Identity gating compares every generated line to
  `base.wav`; a reference recorded in a different room lowers similarity
  for every line forever.

## Phonetically balanced prompt sheet (Hindi/Hinglish)

Read at natural pace, neutral tone, for `base.wav`. The sheet covers all
10 viseme classes: bilabials (प/ब/म), labiodentals (फ/व), dentals
(त/द/थ/स), retroflex (ट/ड/ण), palatals (च/ज/श), velars (क/ग), open
vowels (आ), rounded vowels (ऊ/ओ), spread vowels (ई/ए), and schwa.

1. पानी में पत्थर फेंको तो लहरें बनती हैं, पर क्यों?
2. भौतिकी का मतलब है — हर चीज़ का "क्यों" पूछना।
3. ऊर्जा ना बनती है, ना मिटती है — बस रूप बदलती है।
4. ठीक से देखो: टेबल पर रखी किताब भी धरती को खींच रही है।
5. चुंबक और बिजली — दोनों एक ही सिक्के के दो पहलू हैं।
6. वेग, त्वरण और बल — ये तीनों दोस्त हमेशा साथ चलते हैं।
7. शून्य से शुरू करो, समझ अपने आप गहरी होगी।
8. JEE ka funda simple hai — concept pehle, formula baad mein.
9. Gravity sirf apple girati nahi, moon ko orbit mein rakhti hai.
10. सोचो, समझो, फिर solve करो — यही तरीका है।
11. ओम का नियम: V equals I into R — बस इतना सा।
12. क्षमता, क्षेत्रफल, कोण — कठिन शब्द, आसान ideas.

For each `emotion/<name>.wav`, read sentences 1–2 **in that emotion**:

- `happy` — smiling, warm
- `excited` — fast, energized, higher pitch
- `curious` — questioning, rising intonation
- `surprised` — sharp intake, wide dynamics
- `serious` — slow, low, deliberate

## After recording

1. Drop files into `assets/voices/<character>/…` matching
   `voice_bank.json` paths.
2. Run the validator (above).
3. Preview the whole script's audio without rendering video:

```bash
JV_AUDIO_SOURCE=cloned python3 jvmake.py voice-preview scripts/my_script.json
```

Every generated line passes the speaker-identity gate (cosine similarity
to `base.wav`) and the pronunciation gate (forced-alignment confidence).
A line that fails all retry seeds names itself loudly — record that one
line by hand into `assets/voices/recordings/` and use `RecordedSource`
(see `docs/RUNBOOK.md`).
