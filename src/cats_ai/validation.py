import re
from jsonschema import ValidationError, validate
import json


def extract_json(text: str):
    """
    Extract JSON from raw model output
    """
    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_json(text: str, schema):
    """
    Checks if json scheme is valid
    """

    parsed_json = extract_json(text)

    if parsed_json is None:
        print("JSON decode error")
        print("Raw output:", text)
        return False

    try:
        validate(instance=parsed_json, schema=schema)
    except ValidationError as e:
        print("Schema validation error:", e.message)
        print("Parsed JSON:", parsed_json)
        return False

    return parsed_json
