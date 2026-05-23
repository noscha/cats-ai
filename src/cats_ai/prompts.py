ACCIDENT_ANALYSIS_PROMPT = """
You are analyzing a real-world dashcam traffic video.

Output ONLY valid raw JSON with this schema:

{
  "accident_present": true,
  "confidence": 0.0,
  "scene_description": "...",
  "main_objects": ["..."],
  "risk_factors": ["..."],
  "reasoning": "...",
  "uncertainty": "low"
}

Rules:
- Treat the video as a real-life traffic scenario, not a fictional or imaginary scene.
- accident_present must be true only if a collision, crash, impact, or clear accident is visible.
- accident_present must be false if traffic looks normal, risky but no crash is visible, or the situation is unclear.
- confidence must be between 0.0 and 1.0.
- scene_description should describe only what is visible, in maximum 1 sentence.
- main_objects should list the main visible traffic participants or scene objects.
- risk_factors should list visible risk factors only.
- reasoning should explain why accident_present is true or false, in maximum 1 sentence.
- uncertainty must be one of: "low", "medium", "high".
- Do not invent causes, traffic violations, or unrealistic objects/events that are not visually supported.
- Output raw JSON only. Do not wrap it in Markdown. Do not use ```json code fences.
""".strip()

ACCIDENT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "accident_present": {
            "type": "boolean",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "scene_description": {
            "type": "string",
            "minLength": 1,
        },
        "main_objects": {
            "type": "array",
            "items": {"type": "string"},
        },
        "risk_factors": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {
            "type": "string",
            "minLength": 1,
        },
        "uncertainty": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": [
        "accident_present",
        "confidence",
        "scene_description",
        "main_objects",
        "risk_factors",
        "reasoning",
        "uncertainty",
    ],
    "additionalProperties": False,
}

###############################################################################################################################

ACCIDENT_DETECTION_PROMPT = """
You are analyzing a real-world dashcam traffic video.

Return ONLY valid raw JSON with this schema:

{
  "accident_present": true
}

Rules:
- accident_present must be true only if a collision, crash, impact, or clear accident is visible.
- accident_present must be false if traffic looks normal, risky but no crash is visible, or the situation is unclear.
- Treat the video as a real-life traffic scenario, not a fictional or imaginary scene.
- Do not invent causes, traffic violations, or unrealistic objects/events.
- Output raw JSON only. Do not wrap it in Markdown. Do not use ```json code fences.
""".strip()

ACCIDENT_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "accident_present": {"type": "boolean"},
    },
    "required": ["accident_present"],
    "additionalProperties": False,
}

###############################################################################################################################

ACCIDENT_PREDICTION_PROMPT = """
You are analyzing a real-world dashcam traffic video sequence.

The video intentionally stops before any possible collision occurs.

Your task is to predict whether an accident is likely to happen immediately after the visible footage.

Return ONLY valid raw JSON with this schema:

{
  "accident_present": true
}

Rules:
- accident_present must be true only if the visible traffic behavior strongly suggests an imminent collision or crash.
- accident_present must be false if traffic appears normal, uncertain, or there is insufficient evidence of an imminent accident.
- Base the prediction only on visible driving behavior, trajectories, speed, distance, lane movement, and interactions between vehicles/road users.
- Do not assume hidden events outside the camera view.
- Do not invent causes, traffic violations, or unrealistic events.
- Treat the footage as a real-world traffic scenario.
- Be conservative: if unclear, return false.
- Output raw JSON only.
- Do not wrap output in Markdown.
- Do not use ```json code fences.
""".strip()

ACCIDENT_PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "accident_present": {"type": "boolean"},
    },
    "required": ["accident_present"],
    "additionalProperties": False,
}

###############################################################################################################################

ACCIDENT_ANALYSIS = (ACCIDENT_ANALYSIS_PROMPT, ACCIDENT_ANALYSIS_SCHEMA)
ACCIDENT_DETECTION = (ACCIDENT_DETECTION_PROMPT, ACCIDENT_DETECTION_SCHEMA)
ACCIDENT_PREDICTION = (ACCIDENT_PREDICTION_PROMPT, ACCIDENT_PREDICTION_SCHEMA)


"""
- Focus on near-term risk within the next few seconds after the clip ends.
- Indicators may include unavoidable closing speed, loss of control, dangerous cut-ins, failure to yield, imminent intersection conflict, or pedestrians/cyclists in immediate danger.

{
  "accident_present": true,
  "confidence": 0.91
}

"confidence": {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0
}

or try accident_likely
"""
