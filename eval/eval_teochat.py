import os
import json
import csv
import argparse
import string

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoConfig
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from videollava.model.builder import load_pretrained_model
from videollava.mm_utils import get_model_name_from_path
from videollava.eval.inference import run_inference_single


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


def load_images(image_paths, max_images=None):
    image_paths = [p for p in image_paths if os.path.exists(p)]
    if len(image_paths) == 0:
        raise ValueError("No images found")

    image_paths = sorted(image_paths)

    if max_images is not None and len(image_paths) > max_images:
        # simple uniform subsample only if requested
        idx = torch.linspace(0, len(image_paths) - 1, max_images).long().tolist()
        image_paths = [image_paths[i] for i in idx]

    images = [Image.open(p).convert("RGB") for p in image_paths]
    return images, image_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teochat_path", required=True)
    parser.add_argument("--languagebind_image_path", required=True)
    parser.add_argument("--output_csv", default="results_teochat.csv")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device}!')
    
    model_name = get_model_name_from_path(args.teochat_path)
    
    tokenizer, model, _, context_len = load_pretrained_model(
        model_path=args.teochat_path,
        model_base=None,
        model_name=model_name,
        load_8bit=True,
        load_4bit=False,
        device="cuda",
    )

    # Force-load the towers that were deferred
    if hasattr(model.get_model().get_image_tower(), 'load_model'):
        model.get_model().get_image_tower().load_model()
    
    if hasattr(model.get_model().get_video_tower(), 'load_model'):
        model.get_model().get_video_tower().load_model()
    
    model = model.half().to(device)

    image_processor = AutoImageProcessor.from_pretrained(
        args.languagebind_image_path,
        trust_remote_code=True
    )

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

            image_paths = []
            for rel_path in example["video"]:
                rel_path = rel_path.replace("skyscraper_gdelt_sentinel/", "")
                image_paths.append(os.path.join(args.data_root, rel_path))

            try:
                images, used_paths = load_images(image_paths, max_images=args.max_images)

                torch.cuda.empty_cache()

                out = run_inference_single(
                    model=model,
                    processor=image_processor,
                    tokenizer=tokenizer,
                    inp=prompt,
                    image_paths=images,  # PIL images
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )

                if isinstance(out, tuple):
                    prediction = out[-1]
                else:
                    prediction = out

                prediction = str(prediction).strip()
                prediction = prediction.translate(str.maketrans("", "", string.punctuation))

            except Exception as e:
                prediction = f"ERROR: {e}"

            writer.writerow([
                example_id,
                prompt,
                ground_truth.lower(),
                prediction.lower(),
            ])
            f.flush()
            print(example_id, "|", ground_truth.lower(), "|", prediction.lower())


if __name__ == "__main__":
    main()