#!/usr/bin/env python

import os
import json
import copy
import random
import argparse

import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    VideoLlavaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_json(path):
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    for key in ["data", "samples", "annotations"]:
        if isinstance(data, dict) and key in data:
            return data[key]

    raise ValueError(f"Could not read dataset file: {path}")


def load_data(path, split="train", cache_dir=None):
    if path.endswith(".json") or path.endswith(".jsonl"):
        return load_json(path)

    if os.path.isdir(path):
        candidates = [
            os.path.join(path, f"{split}.json"),
            os.path.join(path, f"{split}.jsonl"),
            os.path.join(path, "train.json"),
            os.path.join(path, "train.jsonl"),
            os.path.join(path, "data.json"),
            os.path.join(path, "data.jsonl"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return load_json(candidate)

        if os.path.exists(os.path.join(path, "dataset_info.json")):
            dataset = load_from_disk(path)
            return dataset[split] if isinstance(dataset, dict) else dataset

    return load_dataset(path, split=split, cache_dir=cache_dir)


def resolve_path(path, data_root=None):
    if os.path.isabs(path):
        return path
    return os.path.join(data_root, path) if data_root else path


def load_image(path):
    return Image.open(path).convert("RGB")


def list_frame_dir(path):
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.lower().endswith(IMAGE_EXTS)
    )


def sample_frames(paths, max_frames):
    if len(paths) <= max_frames:
        return paths

    indices = torch.linspace(0, len(paths) - 1, max_frames).long().tolist()
    return [paths[i] for i in indices]


def sort_by_timestamp(paths, sample):
    timestamps = None

    if "timestamp" in sample and sample["timestamp"]:
        timestamps = sample["timestamp"]

    elif "metadata" in sample and sample["metadata"]:
        metadata = sample["metadata"]
        if isinstance(metadata, list) and len(metadata) > 0 and "timestamp" in metadata[0]:
            timestamps = [m["timestamp"] for m in metadata]

    if timestamps is None or len(timestamps) != len(paths):
        return paths

    return [
        path for path, _ in sorted(
            zip(paths, timestamps),
            key=lambda x: str(x[1]),
        )
    ]


def get_media_paths(sample, data_root=None):
    if "video" in sample:
        video = sample["video"]

        if isinstance(video, list):
            paths = [resolve_path(p, data_root) for p in video]
        else:
            video = resolve_path(video, data_root)
            paths = list_frame_dir(video) if os.path.isdir(video) else [video]

        return sort_by_timestamp(paths, sample)

    if "image" in sample:
        image = sample["image"]

        if isinstance(image, list):
            return [resolve_path(p, data_root) for p in image]

        return [resolve_path(image, data_root)]

    raise ValueError("Sample must contain either 'video' or 'image'.")


def get_question_answer(sample):
    if "conversations" in sample:
        question = None
        answer = None

        for msg in sample["conversations"]:
            role = msg.get("from", msg.get("role", "")).lower()
            text = msg.get("value", msg.get("content", ""))

            if role in ["human", "user"] and question is None:
                question = str(text)

            elif role in ["gpt", "assistant"] and answer is None:
                answer = str(text)

        if question is not None and answer is not None:
            return question, answer

    if "question" in sample and "answer" in sample:
        return str(sample["question"]), str(sample["answer"])

    if "prompt" in sample and "answer" in sample:
        return str(sample["prompt"]), str(sample["answer"])

    if "instruction" in sample and "output" in sample:
        return str(sample["instruction"]), str(sample["output"])

    raise ValueError("Could not find question/answer in sample.")


def clean_question(question):
    return question.replace("times:", "times in chronological order:").strip()


def format_videollava_text(question, answer, eos_token):
    question = clean_question(question)
    question = question.replace(IMAGE_TOKEN, "").strip()

    if VIDEO_TOKEN not in question:
        question = f"{VIDEO_TOKEN}\n{question}"

    prompt = f"USER: {question}\nASSISTANT:"
    full = f"{prompt} {answer.strip()}{eos_token or ''}"

    return prompt, full


def format_gemma3_text(processor, question, answer, num_images):
    question = clean_question(question)
    question = question.replace(VIDEO_TOKEN, "").replace(IMAGE_TOKEN, "").strip()

    user_content = []

    for i in range(num_images):
        user_content.append({"type": "text", "text": f"Image {i + 1}:"})
        user_content.append({"type": "image"})

    user_content.append({"type": "text", "text": question})

    prompt_messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    full_messages = [
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": answer.strip(),
                }
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return prompt, full


class MultimodalDataset(Dataset):
    def __init__(
        self,
        data_name,
        processor,
        model_type,
        max_frames,
        data_split="train",
        data_root=None,
        cache_dir=None,
    ):
        self.data = load_data(
            path=data_name,
            split=data_split,
            cache_dir=cache_dir,
        )
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_type = model_type
        self.max_frames = max_frames
        self.data_root = data_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            sample = self.data[idx]

            paths = get_media_paths(sample, self.data_root)
            paths = sample_frames(paths, self.max_frames)
            images = [load_image(path) for path in paths]

            question, answer = get_question_answer(sample)

            if self.model_type == "videollava":
                prompt, full = format_videollava_text(
                    question=question,
                    answer=answer,
                    eos_token=self.tokenizer.eos_token,
                )

                return {
                    "prompt": prompt,
                    "full": full,
                    "frames": images,
                }

            if self.model_type == "gemma3":
                prompt, full = format_gemma3_text(
                    processor=self.processor,
                    question=question,
                    answer=answer,
                    num_images=len(images),
                )

                return {
                    "prompt": prompt,
                    "full": full,
                    "images": images,
                }

            raise ValueError(f"Unknown model_type: {self.model_type}")

        except Exception as e:
            print(f"Bad sample {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.data) - 1))


class Collator:
    def __init__(self, processor, model_type):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_type = model_type

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError("Use --per_device_train_batch_size 1.")

        item = batch[0]

        if self.model_type == "videollava":
            full_inputs = self.processor(
                text=[item["full"]],
                videos=[item["frames"]],
                return_tensors="pt",
                padding=False,
            )

            prompt_inputs = self.processor(
                text=[item["prompt"]],
                videos=[item["frames"]],
                return_tensors="pt",
                padding=False,
            )

        elif self.model_type == "gemma3":
            full_inputs = self.processor(
                text=[item["full"]],
                images=item["images"],
                return_tensors="pt",
                padding=False,
            )

            prompt_inputs = self.processor(
                text=[item["prompt"]],
                images=item["images"],
                return_tensors="pt",
                padding=False,
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        labels = full_inputs["input_ids"].clone()

        prompt_len = prompt_inputs["input_ids"].shape[1]
        prompt_len = min(prompt_len, labels.shape[1])

        labels[:, :prompt_len] = IGNORE_INDEX

        if "attention_mask" in full_inputs:
            labels[full_inputs["attention_mask"] == 0] = IGNORE_INDEX

        full_inputs["labels"] = labels
        return full_inputs


def find_lora_targets(model):
    linear_classes = [torch.nn.Linear]

    try:
        import bitsandbytes as bnb
        linear_classes.extend([
            bnb.nn.Linear4bit,
            bnb.nn.Linear8bitLt,
        ])
    except Exception:
        pass

    skip_keywords = [
        "vision",
        "visual",
        "projector",
        "multi_modal_projector",
        "mm_projector",
    ]

    names = set()

    for name, module in model.named_modules():
        if any(keyword in name for keyword in skip_keywords):
            continue

        if any(isinstance(module, cls) for cls in linear_classes):
            names.add(name.split(".")[-1])

    names.discard("lm_head")
    return sorted(names)


def load_model(args):
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    quant_config = None

    if args.bits in [4, 8]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_skip_modules=["lm_head"],
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if args.model_type == "videollava":
        model_cls = VideoLlavaForConditionalGeneration
    elif args.model_type == "gemma3":
        model_cls = Gemma3ForConditionalGeneration
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model_cls.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map="auto" if args.bits in [4, 8] else None,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.model_max_length

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    model.config.use_cache = False

    if args.freeze_backbone:
        model.requires_grad_(False)

    if args.bits in [4, 8]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.lora_enable:
        target_modules = find_lora_targets(model)

        print("LoRA targets:", target_modules)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

        model = get_peft_model(model, lora_config)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model, processor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", choices=["videollava", "gemma3"], required=True)
    parser.add_argument("--model_name_or_path", required=True)

    parser.add_argument("--data_name", required=True)
    parser.add_argument("--data_split", default="train")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--cache_dir", default=None)

    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=2048)

    parser.add_argument("--bits", type=int, choices=[4, 8, 16], default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")

    parser.add_argument("--lora_enable", action="store_true")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--trust_remote_code", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    model, processor = load_model(args)

    dataset = MultimodalDataset(
        data_name=args.data_name,
        processor=processor,
        model_type=args.model_type,
        max_frames=args.max_frames,
        data_split=args.data_split,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
    )

    collator = Collator(
        processor=processor,
        model_type=args.model_type,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",

        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,

        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,

        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()