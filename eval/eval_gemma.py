import os
import json
import csv
import argparse
import string
import re
import traceback
import numpy as np

import torch
from PIL import Image
from tqdm import tqdm
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def load_frames(image_paths, max_frames=8):
    image_paths = [p for p in image_paths if os.path.exists(p)]

    if len(image_paths) == 0:
        raise ValueError("No images found")

    image_paths = sorted(image_paths)

    if max_frames is not None and len(image_paths) > max_frames:
        idx = np.linspace(0, len(image_paths) - 1, max_frames).astype(int).tolist()
        image_paths = [image_paths[i] for i in idx]

    return [Image.open(p).convert("RGB") for p in image_paths]


def get_prompt_and_label(example):
    prompt = ""
    label = ""

    for turn in example["conversations"]:
        if turn["from"] == "human":
            prompt = turn["value"]
        elif turn["from"] == "gpt":
            label = turn["value"]

    if prompt == "":
        raise ValueError("No human prompt found")

    return prompt, label


def mostly_numbers_long(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    digits = sum(c.isdigit() for c in text)
    return len(text) > 50 and digits / max(len(text), 1) > 0.5


def is_valid_date_format(text):
    return re.fullmatch(r"\d{4}-\d{2}-\d{2}, \d{4}-\d{2}-\d{2}", text.strip()) is not None


def is_yes_no(text):
    text = text.translate(str.maketrans("", "", string.punctuation)).strip().lower()
    return text in {"yes", "no"}


def is_less_than_5_words(text):
    return len(text.strip().split()) < 5


def has_number(text):
    return any(c.isdigit() for c in text)


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def run_gemma_single(model, processor, prompt, frames, temperature, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image"} for _ in frames]
                + [{"type": "text", "text": prompt}]
            ),
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=text,
        images=frames,
        return_tensors="pt",
    ).to(get_model_device(model))

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    out = processor.decode(generated, skip_special_tokens=True).strip()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", default="google/gemma-3-4b-it")
    parser.add_argument("--lora_ckpt", default=None)
    parser.add_argument("--output_csv", default="results_gemma.csv")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device}!')

    processor = AutoProcessor.from_pretrained(args.model_path)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.lora_ckpt is not None and args.lora_ckpt.strip() != "":
        from peft import PeftModel

        print(f"Loading LoRA checkpoint from {args.lora_ckpt}")
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            is_trainable=False,
        )

    model.eval()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    existing_ids = set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(int(row["id"]))

    file_exists = os.path.exists(args.output_csv)

    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "question", "ground_truth", "prediction"])

        for example in tqdm(data):
            example_id = example["id"]
            if example_id in existing_ids:
                continue

            prompt, ground_truth = get_prompt_and_label(example)
            if "detection" in args.json_path:
                prompt = "This is a sequence of images capturing the same location at different times: <video> Is wildfire occurring in these images? Please answer with only ONE word: either ''Yes'' or ''No'', with no explanation."
            if "classification" in args.json_path:
                prompt = prompt + " Do not include ANY explanation, simply respond with ONE of the four classes with no other words or commentary."
            
            print(prompt)

            image_paths = []
            for rel_path in example["video"]:
                rel_path = rel_path.replace("skyscraper_gdelt_sentinel/", "")
                image_paths.append(os.path.join(args.data_root, rel_path))

            prediction = None

            for attempt in range(args.max_retries):
                torch.cuda.empty_cache()
                try:
                    frames = load_frames(image_paths, max_frames=args.max_frames)

                    torch.cuda.empty_cache()

                    out = run_gemma_single(
                        model=model,
                        processor=processor,
                        prompt=prompt,
                        frames=frames,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )

                    print(f"OUT (attempt {attempt+1}):", out)

                    if isinstance(out, tuple):
                        pred = out[-1]
                    else:
                        pred = out

                    pred = str(pred).strip()

                    bad_output = (
                        pred == "" or
                        "error" in pred.lower() or
                        re.search(r"(?:\d\s+){3,}\d", pred) or
                        mostly_numbers_long(pred)
                    )

                    if "grounding" in args.json_path:
                        bad_output = bad_output or not is_valid_date_format(pred)
                    if "detection" in args.json_path:
                        bad_output = bad_output or not is_yes_no(pred)
                    if "description" in args.json_path:
                        bad_output = bad_output or is_less_than_5_words(pred)
                    if "classification" in args.json_path:
                        bad_output = bad_output or has_number(pred)

                    if not bad_output:
                        prediction = pred
                        break

                except Exception as e:
                    err_msg = str(e).lower()
                    print(f"attempt {attempt+1} failed: {e}")
                    traceback.print_exc()

                    if "probability tensor" in err_msg or "nan" in err_msg or "inf" in err_msg:
                        torch.cuda.empty_cache()
                        continue

                    prediction = f"ERROR: {e}"
                    break

            if prediction is None:
                prediction = "ERROR: max retries exceeded"

            if not prediction.startswith("ERROR:"):
                prediction = prediction.translate(str.maketrans("", "", string.punctuation)).lower()
            else:
                prediction = prediction.lower()

            ground_truth = ground_truth.translate(str.maketrans("", "", string.punctuation)).lower()

            writer.writerow([example_id, prompt, ground_truth, prediction])
            f.flush()

            print(example_id, "|", ground_truth, "|", prediction)


if __name__ == "__main__":
    main()