import torch
from transformers import TrainingArguments, Trainer
from qwen_vl_utils import process_vision_info
import json
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model

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
    output_dir = SFT_PATH / "qwen_carcrash_sft"
    trainer.save_model(str(output_dir))
