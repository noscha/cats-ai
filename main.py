    # %%
    import os
    import json
    from jsonschema import validate, ValidationError
    import re
    import random
    import glob
    from pathlib import Path
    from collections import Counter
    import cv2
    import tempfile

    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    from qwen_vl_utils import process_vision_info
    from peft import LoraConfig, get_peft_model

    # %%
    torch.device("cuda")
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"

    SEED = 42
    random.seed(SEED)
    RNG = random.Random(SEED + 1)

    ROOT = Path("CarCrash/videos")
    NORMAL_DIR = ROOT / "Normal"
    CRASH_DIR = ROOT / "Crash-1500"
    NFRAMES = 50
    N_PER_CLASS_CRASH = 150  # 10% of data
    N_PER_CLASS_NORMAL = 300  # 10% of data

    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    HF_TOKEN = open("token.txt", "r").readline()
    MODEL_OUTPUT_PATH = "model_output"
    SFT_FILE = "carcrash_sft_small.jsonl"
    MAX_NEW_TOKENS = 64  # 400 and above for analysis, 64 for accident detection

    # %%
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

    ACCIDENT_ANALYSIS = (ACCIDENT_ANALYSIS_PROMPT, ACCIDENT_ANALYSIS_SCHEMA)

    # %%
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

    ACCIDENT_DETECTION = (ACCIDENT_DETECTION_PROMPT, ACCIDENT_DETECTION_SCHEMA)

    # %%
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

    # %%
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        token=HF_TOKEN,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
    )

    # %%
    def set_frames(video_path: str, n_frames: int, tmp_dir: Path):
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stem = Path(video_path).stem
        out_path = tmp_dir / f"{stem}_first_{n_frames}.mp4"

        out = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        return str(out_path)


    def get_last_non_accident_frame(video_path: str):

        if "Normal" in video_path:
            # Under 3 frames video pipe breaks
            return RNG.randint(3, 50)

        file = open("CarCrash/videos/Crash-1500.txt")
        content = file.readlines()
        index = int(Path(video_path).stem) - 1
        start = content[index].index("[")
        end = content[index].index("]", start)
        lst = [int(x.strip()) for x in content[index][start + 1 : end].split(",")]

        return lst.index(1)


    def build_messages(
        video_path,
        prompt,
        last_non_accident_frame=float("inf"),
        sft_target=None,
    ):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "nframes": min([NFRAMES, last_non_accident_frame - 1]),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if sft_target is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(sft_target, ensure_ascii=False),
                        }
                    ],
                }
            )

        return messages


    def query(video_path, prompt, crash_masking=False, tmp_dir=None):
        """
        Query a video
        """
        last_non_accident_frame = float("inf")

        if crash_masking:
            last_non_accident_frame = get_last_non_accident_frame(video_path)
            video_path = set_frames(video_path, last_non_accident_frame, tmp_dir)

        messages = build_messages(video_path, prompt, last_non_accident_frame)

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images, videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        videos, video_metadata = zip(*videos)
        videos = list(videos)
        video_metadata = list(video_metadata)
        print(video_metadata)

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response

    # %%
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
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            print("Raw output:", text)
            return False
        except ValidationError as e:
            print("Schema validation error:", e.message)
            print("Parsed JSON:", parsed_json)
            return False

        return parsed_json


    def sample_generator():
        """
        Generator for video files
        """
        for path_str in glob.iglob("**/*.mp4", root_dir=ROOT, recursive=True):
            rel_path = Path(path_str)
            full_path = ROOT / rel_path

            if "Normal" in rel_path.parts:
                label = "normal"
            elif "Crash-1500" in rel_path.parts:
                label = "crash"

            yield full_path, label


    def sample_generator_limited():
        """
        Yield a random balanced subset
        """

        normal_paths = list((ROOT / "Normal").glob("*.mp4"))
        crash_paths = list((ROOT / "Crash-1500").glob("*.mp4"))

        normal_sample = RNG.sample(normal_paths, N_PER_CLASS_NORMAL)
        crash_sample = RNG.sample(crash_paths, N_PER_CLASS_CRASH)

        samples = [(path, "normal") for path in normal_sample] + [
            (path, "crash") for path in crash_sample
        ]

        RNG.shuffle(samples)

        yield from samples


    def trial(prompt_schema_pair, masking):

        tmp_dir = Path(tempfile.mkdtemp(prefix="carcrash_masked_"))
        print(f"Temp folder: {tmp_dir}")

        results = []
        invalid_counter = 0

        for i, (video_path, label) in enumerate(sample_generator_limited(), start=1):
            print(f"\n[{i}] {label} -> {video_path}")

            prompt, schema = prompt_schema_pair
            out = query(str(video_path), prompt, masking, tmp_dir=tmp_dir)
            out = validate_json(out, schema)

            if out:
                pred_accident = out.get("accident_present")

                results.append(
                    {
                        "video": str(video_path),
                        "label": label,
                        "pred_accident": pred_accident,
                        "json": out,
                        "correct_detection": (
                            pred_accident == (True if label == "crash" else False)
                        ),
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

        out_path = MODEL_OUTPUT_PATH / Path("trial_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved to: {out_path.resolve()}")


    def experiment():
        trial(ACCIDENT_PREDICTION, True)

    # %%
    if __name__ == "__main__":
        experiment()
        # query("CarCrash/videos/Normal/000007.mp4", ACCIDENT_PREDICTION_PROMPT, True)

    # %%
    class VideoSFTDataset(Dataset):
        def __init__(self, jsonl_file):
            self.rows = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.rows.append(json.loads(line))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]


    def collate_fn(batch):
        texts = []
        image_inputs_all = []
        video_inputs_all = []
        video_metadata_all = []
        video_kwargs_final = {}

        for row in batch:
            messages = row["messages"]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            images, videos_raw, video_kwargs_raw = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            video_kwargs = video_kwargs_raw or {}
            video_kwargs_final.update(video_kwargs)

            image_inputs_all.extend(images)

            videos, video_metadata = zip(*videos_raw)
            video_inputs_all.extend(list(videos))
            video_metadata_all.extend(list(video_metadata))

            texts.append(text)

        model_inputs = processor(
            text=texts,
            images=image_inputs_all if image_inputs_all else None,
            videos=video_inputs_all if video_inputs_all else None,
            video_metadata=video_metadata_all if video_metadata_all else None,
            padding=True,
            return_tensors="pt",
            **video_kwargs_final,
        )

        # TODO ONlY for pipeline testing, later do masking
        labels = model_inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels

        """
        labels = model_inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[:, :prompt_len] = -100
        model_inputs["labels"] = labels
        """

        return model_inputs


    def sft():
        """
        LoRA SFT on Qwen.
        """
        model.config.use_cache = False

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        fine_tuned_model = get_peft_model(model, lora_config)
        fine_tuned_model.print_trainable_parameters()

        train_dataset = VideoSFTDataset(SFT_FILE)

        training_args = TrainingArguments(
            output_dir="./qwen_carcrash_sft",
            per_device_train_batch_size=1,  # TODO: lookup if increasing is useful in video setting
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=1,
            save_steps=20,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            fp16=False,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=fine_tuned_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
        )

        trainer.train()
        trainer.save_model("./qwen_carcrash_sft/final")


