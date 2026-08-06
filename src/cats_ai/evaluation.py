import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from cats_ai.config import MODEL_OUTPUT_PATH, seed_everything, ROOT
from cats_ai.inference import query
from cats_ai.model import load_model_and_processor
from cats_ai.prompts import ACCIDENT_PREDICTION, ACCIDENT_ANALYSIS, ACCIDENT_DETECTION
from cats_ai.sampling import sample_generator
from cats_ai.validation import validate_json


def trial(root, prompt_schema_pair, masking, sample_fn=sample_generator):

    tag = datetime.now().strftime("%Y-%m-%d_%H:%M")
    (MODEL_OUTPUT_PATH / tag).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="carcrash_masked_"))

    results = []
    invalid_counter = 0

    prompt, schema = prompt_schema_pair
    model, processor = load_model_and_processor()

    for i, (video_path, label) in enumerate(sample_fn(root), start=1):
        print(f"\n[{i}] {label} -> {video_path}", flush=True)

        out = query(
            str(video_path),
            prompt,
            model=model,
            processor=processor,
            crash_masking=masking,
            tmp_dir=tmp_dir,
        )
        out = validate_json(out, schema)

        if out:
            pred_accident = out.get("accident_present")

            results.append(
                {
                    "video": str(video_path),
                    "label": label,
                    "pred_accident": pred_accident,
                    "json": out,
                    "correct_detection": (pred_accident == (label == "crash")),
                }
            )

        else:
            print(f"{i} is invalid")
            invalid_counter += 1

    valid = len(results) / (len(results) + invalid_counter)
    acc = sum(r["correct_detection"] for r in results) / len(results)

    print("\n=== SUMMARY ===")
    print(f"Valid predictions: {valid * 100}%")
    print(f"Detection accuracy: {acc:.3f}")

    print("\nBreakdown:")
    print(Counter((r["label"], r["pred_accident"]) for r in results))

    out_path = MODEL_OUTPUT_PATH / Path("trial_results_.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {out_path.resolve()}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Deleted temp folder: {tmp_dir}")


def experiment():
    seed_everything(deterministic=True)
    trial(
        ROOT, ACCIDENT_DETECTION, False, sample_generator
    )  # masking must be true for prediction
