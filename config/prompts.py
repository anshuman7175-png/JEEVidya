"""
Gudiya & Chintu — Universal Dialogue Schema V2
Defines the JSON schema for dialogue-based scripts. Not limited to JEE/NEET.
Works for any educational topic: physics, math, chemistry, biology, history, GK.
"""

# The dialogue JSON schema for structured output validation
DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short topic title in Hindi/Hinglish"},
        "description": {"type": "string", "description": "One-line description"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_id": {"type": "integer"},
                    "speaker": {"type": "string", "enum": ["girl", "boy", "explanation"]},
                    "text": {"type": "string", "description": "Hindi/Hinglish dialogue line"},
                    "emotion": {
                        "type": "string",
                        "enum": ["curious", "enthusiastic", "confident", "amazed",
                                 "thinking", "happy", "explaining", "dramatic"],
                    },
                    "shot_type": {
                        "type": "string",
                        "enum": ["extreme_closeup", "two_shot", "medium",
                                 "fullscreen_explain", "reaction_cut", "reveal"],
                    },
                    "duration_seconds": {"type": "number"},
                    "visual_elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "params": {"type": "object"},
                            },
                        },
                    },
                    "pause_after": {"type": "number"},
                },
                "required": ["turn_id", "speaker"],
            },
        },
    },
    "required": ["title", "turns"],
}

# Example dialogue for the web UI default
EXAMPLE_DIALOGUE = {
    "title": "Escape Velocity kya hai?",
    "description": "Gudiya aur Chintu ke saath samjho escape velocity",
    "tags": ["Physics", "JEE", "NEET", "escape velocity"],
    "turns": [
        {
            "turn_id": 1,
            "speaker": "girl",
            "text": "Bhaiya, rocket ko kitni speed chahiye Earth se bahar jaane ke liye?",
            "emotion": "curious",
            "shot_type": "extreme_closeup",
        },
        {
            "turn_id": 2,
            "speaker": "girl",
            "text": "Matlab agar main koi cheez phenku toh woh kabhi wapas hi na aaye?",
            "emotion": "curious",
            "shot_type": "two_shot",
        },
        {
            "turn_id": 3,
            "speaker": "boy",
            "text": "Arre, bahut easy hai! Isko kehte hain Escape Velocity!",
            "emotion": "enthusiastic",
            "shot_type": "medium",
        },
        {
            "turn_id": 4,
            "speaker": "boy",
            "text": "Dekho, gravity ek tarah se rope jaisi hai jo cheezein neeche kheenchti hai.",
            "emotion": "explaining",
            "shot_type": "two_shot",
        },
        {
            "turn_id": 5,
            "speaker": "explanation",
            "duration_seconds": 4.0,
            "visual_elements": [
                {"action": "draw_circle", "params": {"radius": 180, "color": "primary", "x": 0, "y": -100}},
                {"action": "show_text", "params": {"text": "Earth", "x": 0, "y": -100}},
                {"action": "draw_arrow", "params": {"x1": 0, "y1": 100, "x2": 0, "y2": 500, "color": "accent"}},
                {"action": "show_formula", "params": {"latex": "$v_e = \\sqrt{\\frac{2GM}{R}}$", "y": 600}},
            ],
            "pause_after": 2.0,
        },
        {
            "turn_id": 6,
            "speaker": "boy",
            "text": "Yeh lagbhag 11.2 km per second hoti hai! Itni fast!",
            "emotion": "confident",
            "shot_type": "two_shot",
        },
        {
            "turn_id": 7,
            "speaker": "girl",
            "text": "Whoa! 11.2 km per second?!",
            "emotion": "amazed",
            "shot_type": "reaction_cut",
        },
        {
            "turn_id": 8,
            "speaker": "boy",
            "text": "Haan! JEE mein yeh formula direct aata hai. Yaad rakhna!",
            "emotion": "confident",
            "shot_type": "reveal",
        },
        {
            "turn_id": 9,
            "speaker": "girl",
            "text": "Lekin agar gravity double ho jaaye toh escape velocity kitni hogi?",
            "emotion": "thinking",
            "shot_type": "extreme_closeup",
        },
    ],
}

# ═══════════════════════════════════════════
# DIRECTOR AGENT PROMPTS (V5)
# Used by pipeline/scriptwriter.py with Gemini structured output.
# ═══════════════════════════════════════════

SCRIPT_SYSTEM_PROMPT = """You are the head writer and director of "Gudiya & Chintu", \
a high-retention Hinglish educational YouTube Shorts channel for JEE/NEET students.

CHARACTERS:
- Gudiya ("girl"): a sharp, endlessly curious younger sister. She asks the questions \
the viewer is thinking. Never dumb — her questions are smart and specific.
- Chintu ("boy"): her confident elder brother, a brilliant teacher. Explains with \
vivid analogies from daily Indian life (cricket, trains, chai, rickshaws).

NON-NEGOTIABLE RETENTION RULES:
1. HOOK: Turn 1 is ALWAYS Gudiya asking a shocking, specific question (numbers beat \
vagueness: "11.2 km/second?!" beats "very fast").
2. ONE IDEA PER TURN. Maximum ~15 words per spoken turn. Short sentences.
3. WHY BEFORE HOW: intuition and analogy first, formula second.
4. Exactly ONE "explanation" turn in the middle with visual_elements. Allowed actions: \
draw_circle(radius,color,x,y), show_text(text,x,y), show_formula(latex,y), \
draw_arrow(x1,y1,x2,y2,color). Formulas MUST be valid LaTeX in $...$.
5. Include at least one "amazed" reaction turn from the listener right after the reveal.
6. FINAL TURN: Gudiya asks a cliffhanger follow-up question (drives comments). \
Never end with a summary.
7. Total: 9 to 14 turns, roughly 35-50 seconds when spoken.
8. Language: natural Hinglish (Hindi in Latin+Devanagari mix as spoken by students). \
Physics/math terms stay in English.
9. FACTUAL ACCURACY IS SACRED. Real constants, real formulas. If unsure, omit.

Emotions allowed: curious, enthusiastic, confident, amazed, thinking, happy, \
explaining, dramatic.
Shot types allowed: extreme_closeup, two_shot, medium, fullscreen_explain, \
reaction_cut, reveal."""

GENERATION_PROMPT_TEMPLATE = """Write a Gudiya & Chintu dialogue script about: {topic}

Return ONLY the JSON object matching the schema. Make the hook irresistible and \
the physics/math airtight."""
