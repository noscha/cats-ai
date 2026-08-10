import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from cats_ai.config import MODEL_OUTPUT_PATH, ROOT, seed_everything
from cats_ai.inference import query_video
from cats_ai.model import load_model_and_processor
from cats_ai.prompts import VIDEO_SOURCE
from cats_ai.sampling import sample_generator
from cats_ai.validation import validate_json


def trial_accident(root, prompt_schema_pair, masking, sample_fn=sample_generator):

    tmp_dir = Path(tempfile.mkdtemp(prefix="carcrash_masked_"))

    results = []
    invalid_counter = 0

    prompt, schema = prompt_schema_pair
    model, processor = load_model_and_processor()

    for i, (video_path, label) in enumerate(sample_fn(root), start=1):
        print(f"\n[{i}] {label} -> {video_path}", flush=True)

        out, last_non_accident_frame = query_video(
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
                    "last_non_accident_frame": last_non_accident_frame,  # inf if not used
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

    # save results
    tag = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    out_path = MODEL_OUTPUT_PATH / ("accident_" + tag)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / Path("trial_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {out_file.resolve()}")

    # delete temp dir for masked vids
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Deleted temp folder: {tmp_dir}")


def trial_source(root, prompt_schema_pair, sample_fn=sample_generator):

    results = []
    invalid_counter = 0

    prompt, schema = prompt_schema_pair
    model, processor = load_model_and_processor()

    for i, (video_path, label) in enumerate(sample_fn(root), start=1):
        print(f"\n[{i}] {label} -> {video_path}", flush=True)

        out = query_video(
            str(video_path),
            prompt,
            model=model,
            processor=processor,
            crash_masking=False,
            tmp_dir=None,
        )
        out = validate_json(out, schema)

        if out:
            pred_source = out.get("source_type")

            results.append(
                {
                    "video": str(video_path),
                    "label": label,
                    "pred_source": pred_source,
                    "json": out,
                    "correct_source": (pred_source == (label == "crash")),
                }
            )

        else:
            print(f"{i} is invalid")
            invalid_counter += 1

    valid = len(results) / (len(results) + invalid_counter)
    acc = sum(r["correct_source"] for r in results) / len(results)

    print("\n=== SUMMARY ===")
    print(f"Valid predictions: {valid * 100}%")
    print(f"Detection accuracy: {acc:.3f}")

    print("\nBreakdown:")
    print(Counter((r["label"], r["pred_source"]) for r in results))

    # save results
    tag = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    out_path = MODEL_OUTPUT_PATH / ("source_" + tag)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / Path("trial_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {out_file.resolve()}")


def experiment_accident(prompt_schema_pair, masking):
    seed_everything(deterministic=True)
    trial_accident(
        ROOT, prompt_schema_pair, masking, sample_generator
    )  # masking=true for prediction


def experiment_source():
    seed_everything(deterministic=True)
    trial_source(ROOT, VIDEO_SOURCE, sample_generator)